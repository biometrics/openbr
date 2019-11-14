#include <openbr/openbr_plugin.h>
#include "openbr/plugins/openbr_internal.h"
#include <openbr/core/qtutils.h>

#include <QFutureSynchronizer>
#include <QtConcurrentRun>
#include <QMutexLocker>
#include <QWaitCondition>

#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace br;

class lmdbGallery : public Gallery
{
    Q_OBJECT
    Q_PROPERTY(bool remap READ get_remap WRITE set_remap RESET reset_remap STORED false)
    BR_PROPERTY(bool, remap, true)
    Q_PROPERTY(int cacheLimit READ get_cacheLimit WRITE set_cacheLimit RESET reset_cacheLimit STORED false)
    BR_PROPERTY(int, cacheLimit, 10000)

    TemplateList readBlock(bool *done)
    {
        *done = false;
        if (!initialized) {
            db = QSharedPointer<caffe::db::DB>(caffe::db::GetDB("lmdb"));
            db->Open(file.name.toStdString(),caffe::db::READ);
            cursor = QSharedPointer<caffe::db::Cursor>(db->NewCursor());
            initialized = true;
        }

        caffe::Datum datum;
        datum.ParseFromString(cursor->value());

        cv::Mat img;
        if (datum.encoded()) {
            img = caffe::DecodeDatumToCVMatNative(datum);
        }
        else {
            // create output image of appropriate size
            img.create(datum.height(), datum.width(), CV_MAKETYPE(CV_8U, datum.channels()));
            // copy matrix data from datum.
            for (int h = 0; h < datum.height(); ++h) {
                uchar* ptr = img.ptr<uchar>(h);
                int img_index = 0;
                for (int w = 0; w < datum.width(); ++w) {
                    for (int c = 0; c < datum.channels(); ++c) {
                        int datum_index = (c * datum.height() + h) * datum.width() + w;
                        ptr[img_index++] = (unsigned char)datum.data()[datum_index];
                    }
                }
            }
        }

        // We acquired the image data, now decode filename from db key
        QString baseKey = cursor->key().c_str();

        int idx = baseKey.indexOf("_");
        if (idx != -1)
            baseKey = baseKey.right(baseKey.size() - idx - 1);

        TemplateList output;
        output.append(Template(img));
        output.last().file.name = baseKey;
        output.last().file.set("Label", datum.label());

        cursor->Next();

        if (!cursor->valid())
            *done = true;

        return output;
    }

    bool initialized;
    QSharedPointer<caffe::db::DB> db;
    QSharedPointer<caffe::db::Cursor> cursor;

    QFutureSynchronizer<void> aThread;
    QMutex dataLock;
    QWaitCondition dataWait;

    bool should_end;
    TemplateList data;

    QHash<QString, int> observedLabels;

    static void commitLoop(lmdbGallery * base)
    {
        QSharedPointer<caffe::db::Transaction> txn;

        int total_count = 0;

        // Acquire the lock
        QMutexLocker lock(&base->dataLock);

        while (true) {
            // wait for data, or end signal
            while(base->data.isEmpty() && !base->should_end)
                base->dataWait.wait(&base->dataLock);

            // If should_end, but there is still data, we need another commit
            // round
            if (base->should_end && base->data.isEmpty())
                break;

            txn = QSharedPointer<caffe::db::Transaction>(base->db->NewTransaction());

            TemplateList working = base->data;
            base->data.clear();

            // no longer blocking dataLock
            lock.unlock();

            foreach(const Template &t, working) {
                // add current image to transaction
                caffe::Datum datum;

                const cv::Mat &m = t.m();
                if (m.depth() == CV_32F) {
                    datum.set_channels(m.channels());
                    datum.set_height(m.rows);
                    datum.set_width(m.cols);
                    for (int i=0; i<m.channels(); i++) // Follow Caffe's channel-major ordering convention
                        for (int j=0; j<m.rows; j++)
                            for (int k=0; k<m.cols; k++)
                                datum.add_float_data(m.ptr<float>(j)[k*m.channels()+i]);
                } else {
                    caffe::CVMatToDatum(m, &datum);
                }

                QVariant base_label = t.file.value("Label");
                QString label_str = base_label.toString();

                if (!base->observedLabels.contains(label_str)) {
                    if (base->remap) base->observedLabels.insert(label_str, base->observedLabels.size());
                    else             base->observedLabels.insert(label_str, label_str.toInt());
                }

                datum.set_label(base->observedLabels[label_str]);

                std::string out;
                datum.SerializeToString(&out);

                char key_cstr[256];
                int len = snprintf(key_cstr, 256, "%08d_%s", total_count, qPrintable(t.file.name));
                txn->Put(std::string(key_cstr, len), out);

                total_count++;
            }

            txn->Commit();
            lock.relock();
        }
    }

    void write(const Template &t)
    {
        if (!initialized) {
            db = QSharedPointer<caffe::db::DB> (caffe::db::GetDB("lmdb"));
            db->Open(file.name.toStdString(), caffe::db::NEW);
            observedLabels.clear();
            initialized = true;
            should_end = false;
            // fire thread
            aThread.clearFutures();
            aThread.addFuture(QtConcurrent::run(lmdbGallery::commitLoop, this));
        }

        QMutexLocker lock(&dataLock);
        data.append(t);
        dataWait.wakeAll();

        if (cacheLimit != -1 && data.size() > cacheLimit)
            QThread::msleep(1);
    }

    ~lmdbGallery()
    {
        if (initialized) {
            QMutexLocker lock(&dataLock);
            should_end = true;
            dataWait.wakeAll();
            lock.unlock();

            aThread.waitForFinished();
        }
    }
    
    
    void init()
    {
        initialized = false;
        should_end = false;
    }
};

BR_REGISTER(Gallery, lmdbGallery)


#include "gallery/lmdb.moc"

