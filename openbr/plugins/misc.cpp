/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <QtNetwork>
#include <QElapsedTimer>
#include <QRegularExpression>
#include <opencv2/highgui/highgui.hpp>
#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/qtutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Applies br::Format to br::Template::file::name and appends results.
 * \author Josh Klontz \cite jklontz
 */
class OpenTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        if (src.empty()) {
            if (Globals->verbose)
                qDebug("Opening %s", qPrintable(src.file.flat()));

            // Read from disk otherwise
            foreach (const File &file, src.file.split()) {
                QScopedPointer<Format> format(Factory<Format>::make(file));
                Template t = format->read();
                if (t.isEmpty())
                    qWarning("Can't open %s from %s", qPrintable(file.flat()), qPrintable(QDir::currentPath()));
                dst.append(t);
                dst.file.append(t.file.localMetadata());
            }
            dst.file.set("FTO", dst.isEmpty());
        } else {
            // Propogate or decode existing matricies
            foreach (const Mat &m, src) {
                if (((m.rows > 1) && (m.cols > 1)) || (m.type() != CV_8UC1)) dst += m;
                else                                                         dst += imdecode(src.m(), IMREAD_UNCHANGED);
            }
        }
    }
};

BR_REGISTER(Transform, OpenTransform)

/*!
 * \ingroup transforms
 * \brief Decodes images
 * \author Josh Klontz \cite jklontz
 */
class DecodeTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.append(cv::imdecode(src.m(), IMREAD_UNCHANGED));
    }
};

BR_REGISTER(Transform, DecodeTransform)

/*!
 * \ingroup transforms
 * \brief Downloads an image from a URL
 * \author Josh Klontz \cite jklontz
 */
class DownloadTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_ENUMS(Mode)

public:
    enum Mode { Permissive,
                Encoded,
                Decoded };
private:
    BR_PROPERTY(Mode, mode, Encoded)

    mutable QNetworkAccessManager nam;
    mutable QMutex namLock;

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        QString url = src.file.get<QString>("URL").simplified();
        if (url.isEmpty())
            return;

        if (url.startsWith("file://"))
            url = url.mid(7);

        QIODevice *device = NULL;
        if (QFileInfo(url).exists()) {
            device = new QFile(url);
            device->open(QIODevice::ReadOnly);
        } else {
            namLock.lock();
            QNetworkReply *reply = nam.get(QNetworkRequest(url));
            namLock.unlock();

            reply->waitForReadyRead(-1);
            while (!reply->isFinished())
                QCoreApplication::processEvents();

            if (reply->error() != QNetworkReply::NoError) {
                qDebug() << reply->errorString() << url;
                reply->deleteLater();
            } else {
                device = reply;
            }
        }

        if (!device)
            return;

        const QByteArray data = device->readAll();
        delete device;
        device = NULL;

        Mat encoded(1, data.size(), CV_8UC1, (void*)data.data());
        if (mode == Permissive) {
            dst += encoded;
        } else {
            Mat decoded = imdecode(encoded, IMREAD_UNCHANGED);
            if (!decoded.empty())
                dst += (mode == Encoded) ? encoded : decoded;
        }
    }
};

BR_REGISTER(Transform, DownloadTransform)

/*!
 * \ingroup transforms
 * \brief Prints the template's file to stdout or stderr.
 * \author Josh Klontz \cite jklontz
 */
class PrintTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(bool error READ get_error WRITE set_error RESET reset_error)
    Q_PROPERTY(bool data READ get_data WRITE set_data RESET reset_data)
    BR_PROPERTY(bool, error, true)
    BR_PROPERTY(bool, data, false)

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        const QString nameString = src.file.flat();
        const QString dataString = data ? OpenCVUtils::matrixToString(src)+"\n" : QString();
        QStringList matricies;
        foreach (const Mat &m, src)
            matricies.append(QString::number(m.rows) + "x" + QString::number(m.cols) + "_" + OpenCVUtils::typeToString(m));
        fprintf(error ? stderr : stdout, "%s\n  %s\n%s", qPrintable(nameString), qPrintable(matricies.join(",")), qPrintable(dataString));
    }
};

BR_REGISTER(Transform, PrintTransform)

/*!
 * \ingroup transforms
 * \brief Checks the template for NaN values.
 * \author Josh Klontz \cite jklontz
 */
class CheckTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    static int count;
    int index;

 public:
    CheckTransform() : index(count++) {}

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        foreach (const Mat &m, src) {
            Mat fm;
            m.convertTo(fm, CV_32F);
            const int elements = fm.rows * fm.cols * fm.channels();
            const float *data = (const float*)fm.data;
            for (int i=0; i<elements; i++)
                if (data[i] != data[i])
                    qFatal("%s NaN check %d failed!", qPrintable(src.file.flat()), index);
        }
    }
};

int CheckTransform::count = 0;

BR_REGISTER(Transform, CheckTransform)

/*!
 * \ingroup transforms
 * \brief Sets the template's matrix data to the br::File::name.
 * \author Josh Klontz \cite jklontz
 */
class NameTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        QByteArray name = src.file.baseName().toLocal8Bit();
        dst = Mat(1, name.size(), CV_8UC1);
        memcpy(dst.m().data, name.data(), name.size());
    }
};

BR_REGISTER(Transform, NameTransform)

/*!
 * \ingroup transforms
 * \brief A no-op transform.
 * \see DiscardTransform FirstTransform RestTransform RemoveTransform
 * \author Josh Klontz \cite jklontz
 */
class IdentityTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;
    }
};

BR_REGISTER(Transform, IdentityTransform)

/*!
 * \ingroup transforms
 * \brief Removes all template's matrices.
 * \see IdentityTransform FirstTransform RestTransform RemoveTransform
 * \author Josh Klontz \cite jklontz
 */
class DiscardTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
    }
};

BR_REGISTER(Transform, DiscardTransform)

/*!
 * \ingroup transforms
 * \brief Removes all but the first matrix from the template.
 * \see IdentityTransform DiscardTransform RestTransform RemoveTransform
 * \author Josh Klontz \cite jklontz
 */
class FirstTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        // AggregateFrames will leave the Template empty
        // if it hasn't filled up the buffer
        // so we gotta anticipate an empty Template
        if (src.empty()) return;
        dst.file = src.file;
        dst = src.m();
    }
};

BR_REGISTER(Transform, FirstTransform)

/*!
 * \ingroup transforms
 * \brief Removes the first matrix from the template.
 * \see IdentityTransform DiscardTransform FirstTransform RemoveTransform
 * \author Josh Klontz \cite jklontz
 */
class RestTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.removeFirst();
    }
};

BR_REGISTER(Transform, RestTransform)

/*!
 * \ingroup transforms
 * \brief Removes the matrix from the template at the specified index.
 * \author Josh Klontz \cite jklontz
 * \see IdentityTransform DiscardTransform FirstTransform RestTransform
 */
//! [example_transform]
class RemoveTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int index READ get_index WRITE set_index RESET reset_index STORED false)
    BR_PROPERTY(int, index, 0)

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.removeAt(index);
    }
};

BR_REGISTER(Transform, RemoveTransform)
//! [example_transform]

/*!
 * \ingroup transforms
 * \brief Rename metadata key
 * \author Josh Klontz \cite jklontz
 */
class RenameTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString find READ get_find WRITE set_find RESET reset_find STORED false)
    Q_PROPERTY(QString replace READ get_replace WRITE set_replace RESET reset_replace STORED false)
    BR_PROPERTY(QString, find, "")
    BR_PROPERTY(QString, replace, "")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        if (dst.localKeys().contains(find)) {
            dst.set(replace, dst.value(find));
            dst.remove(find);
        }
    }
};

BR_REGISTER(Transform, RenameTransform)

/*!
 * \ingroup transforms
 * \brief Rename first found metadata key
 * \author Josh Klontz \cite jklontz
 */
class RenameFirstTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList find READ get_find WRITE set_find RESET reset_find STORED false)
    Q_PROPERTY(QString replace READ get_replace WRITE set_replace RESET reset_replace STORED false)
    BR_PROPERTY(QStringList, find, QStringList())
    BR_PROPERTY(QString, replace, "")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        foreach (const QString &key, find)
            if (dst.localKeys().contains(key)) {
                dst.set(replace, dst.value(key));
                dst.remove(key);
                break;
            }
    }
};

BR_REGISTER(Transform, RenameFirstTransform)

/*!
 * \ingroup transforms
 * \brief Change the br::Template::file extension
 * \author Josh Klontz \cite jklontz
 */
class AsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString extension READ get_extension WRITE set_extension RESET reset_extension STORED false)
    BR_PROPERTY(QString, extension, "")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        dst.name = dst.name.left(dst.name.lastIndexOf('.')+1) + extension;
    }
};

BR_REGISTER(Transform, AsTransform)

/*!
 * \ingroup transforms
 * \brief Apply the input regular expression to the value of inputProperty, store the matched portion in outputProperty.
 * \author Charles Otto \cite caotto
 */
class RegexPropertyTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString regexp READ get_regexp WRITE set_regexp RESET reset_regexp STORED false)
    Q_PROPERTY(QString inputProperty READ get_inputProperty WRITE set_inputProperty RESET reset_inputProperty STORED false)
    Q_PROPERTY(QString outputProperty READ get_outputProperty WRITE set_outputProperty RESET reset_outputProperty STORED false)
    BR_PROPERTY(QString, regexp, "(.*)")
    BR_PROPERTY(QString, inputProperty, "name")
    BR_PROPERTY(QString, outputProperty, "Label")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        QRegularExpression re(regexp);
        QRegularExpressionMatch match = re.match(dst.get<QString>(inputProperty));
        if (!match.hasMatch())
            qFatal("Unable to match regular expression \"%s\" to base name \"%s\"!", qPrintable(regexp), qPrintable(dst.get<QString>(inputProperty)));
        dst.set(outputProperty, match.captured(match.lastCapturedIndex()));
    }
};

BR_REGISTER(Transform, RegexPropertyTransform)

/*!
 * \ingroup transforms
 * \brief Store the last matrix of the input template as a metadata key with input property name.
 * \author Charles Otto \cite caotto
 */
class SaveMatTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(QString propName READ get_propName WRITE set_propName RESET reset_propName STORED false)
    BR_PROPERTY(QString, propName, "")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.file.set(propName, QVariant::fromValue(dst.m()));
    }
};
BR_REGISTER(Transform, SaveMatTransform)

/*!
 * \ingroup transforms
 * \brief Set the last matrix of the input template to a matrix stored as metadata with input propName.
 *
 * Also removes the property from the templates metadata after restoring it.
 *
 * \author Charles Otto \cite caotto
 */
class RestoreMatTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString propName READ get_propName WRITE set_propName RESET reset_propName STORED false)
    BR_PROPERTY(QString, propName, "")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        if (dst.file.contains(propName)) {
            dst.clear();
            dst.m() = dst.file.get<cv::Mat>(propName);
            dst.file.remove(propName);
        }
    }
};
BR_REGISTER(Transform, RestoreMatTransform)


/*!
 * \ingroup transforms
 * \brief Incrementally output templates received to a gallery, based on the current filename
 * When a template is received in projectUpdate for the first time since a finalize, open a new gallery based on the
 * template's filename, and the galleryFormat property.
 * Templates received in projectUpdate will be output to the gallery with a filename combining their original filename and
 * their FrameNumber property, with the file extension specified by the fileFormat property.
 * \author Charles Otto \cite caotto
 */
class IncrementalOutputTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString galleryFormat READ get_galleryFormat WRITE set_galleryFormat RESET reset_galleryFormat STORED false)
    Q_PROPERTY(QString fileFormat READ get_fileFormat WRITE set_fileFormat RESET reset_fileFormat STORED false)
    BR_PROPERTY(QString, galleryFormat, "")
    BR_PROPERTY(QString, fileFormat, ".png")

    bool galleryUp;

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        if (src.empty())
            return;

        if (!galleryUp) {
            QFileInfo finfo(src[0].file.name);
            QString galleryName = finfo.baseName() + galleryFormat;

            writer = QSharedPointer<Gallery> (Factory<Gallery>::make(galleryName));
            galleryUp = true;
        }

        dst = src;
        int idx =0;
        foreach(const Template & t, src) {
            if (t.empty())
                continue;

            // Build the output filename for this template
            QFileInfo finfo(t.file.name);
            QString outputName = finfo.baseName() +"_" + t.file.get<QString>("FrameNumber") + "_" + QString::number(idx)+ fileFormat;

            idx++;
            Template out = t;
            out.file.name = outputName;
            writer->write(out);
        }
    }

    void train(const TemplateList& data)
    {
        (void) data;
    }

    // Drop the current gallery.
    void finalize(TemplateList & data)
    {
        (void) data;
        galleryUp = false;
    }

    QSharedPointer<Gallery> writer;
public:
    IncrementalOutputTransform() : TimeVaryingTransform(false,false) {galleryUp = false;}
};

BR_REGISTER(Transform, IncrementalOutputTransform)

class EventTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString eventName READ get_eventName WRITE set_eventName RESET reset_eventName STORED false)
    BR_PROPERTY(QString, eventName, "")

    TemplateEvent event;

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        event.pulseSignal(dst);
    }

    TemplateEvent * getEvent(const QString & name)
    {
        return name == eventName ? &event : NULL;
    }
};
BR_REGISTER(Transform, EventTransform)


class GalleryOutputTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString outputString READ get_outputString WRITE set_outputString RESET reset_outputString STORED false)
    BR_PROPERTY(QString, outputString, "")

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        if (src.empty())
            return;
        dst = src;
        writer->writeBlock(dst);
    }

    void train(const TemplateList& data)
    {
        (void) data;
    }
    ;
    void init()
    {
        writer = QSharedPointer<Gallery>(Gallery::make(outputString));
    }

    QSharedPointer<Gallery> writer;
public:
    GalleryOutputTransform() : TimeVaryingTransform(false,false) {}
};

BR_REGISTER(Transform, GalleryOutputTransform)

class ProgressCounterTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(int totalTemplates READ get_totalTemplates WRITE set_totalTemplates RESET reset_totalTemplates STORED false)
    BR_PROPERTY(int, totalTemplates, 1)

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst = src;

        qint64 elapsed = timer.elapsed();

        if (!dst.empty()) {
            Globals->currentProgress = dst.last().file.get<qint64>("progress",0);
            dst.last().file.remove("progress");
            Globals->currentStep++;
        }

        // updated every second
        if (elapsed > 1000) {
            Globals->printStatus();
            timer.start();
        }

        return;
    }

    void train(const TemplateList& data)
    {
        (void) data;
    }

    void finalize(TemplateList & data)
    {
        (void) data;
        float p = br_progress();
        qDebug("%05.2f%%  ELAPSED=%s  REMAINING=%s  COUNT=%g  \r", p*100, QtUtils::toTime(Globals->startTime.elapsed()/1000.0f).toStdString().c_str(), QtUtils::toTime(0).toStdString().c_str(), Globals->currentStep);
    }

    void init()
    {
        timer.start();
        Globals->currentStep = 0;
    }

public:
    ProgressCounterTransform() : TimeVaryingTransform(false,false) {}
    QElapsedTimer timer;
};

BR_REGISTER(Transform, ProgressCounterTransform)


class OutputTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString outputString READ get_outputString WRITE set_outputString RESET reset_outputString STORED false)
    // names of mem galleries containing filelists we need.
    Q_PROPERTY(QString targetName READ get_targetName WRITE set_targetName RESET reset_targetName STORED false)
    Q_PROPERTY(QString queryName  READ get_queryName WRITE set_queryName RESET reset_queryName STORED false)
    Q_PROPERTY(bool transposeMode  READ get_transposeMode WRITE set_transposeMode RESET reset_transposeMode STORED false)

    BR_PROPERTY(QString, outputString, "")
    BR_PROPERTY(QString, targetName, "")
    BR_PROPERTY(QString, queryName, "")

    BR_PROPERTY(bool,transposeMode, false)
    ;

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst = src;

        if (src.empty())
            return;

        // we received a template, which is the next row/column in order
        foreach(const Template & t, dst) {
            for (int i=0; i < t.m().cols; i++)
            {
                output->setRelative(t.m().at<float>(0, i), currentRow, currentCol);

                // row-major input
                if (!transposeMode)
                    currentCol++;
                // col-major input
                else
                    currentRow++;
            }
            // filled in a row, advance to the next, reset column position
            if (!transposeMode) {
                currentRow++;
                currentCol = 0;
            }
            // filled in a column, advance, reset row
            else {
                currentCol++;
                currentRow = 0;
            }
        }

        bool blockDone = false;
        // In direct mode, we don't buffer rows
        if (!transposeMode)
        {
            currentBlockRow++;
            blockDone = true;
        }
        // in transpose mode, we buffer 100 cols before writing the block
        else if (currentCol == bufferedSize)
        {
            currentBlockCol++;
            blockDone = true;
        }
        else return;

        if (blockDone)
        {
            // set the next block, only necessary if we haven't buffered the current item
            output->setBlock(currentBlockRow, currentBlockCol);
            currentRow = 0;
            currentCol = 0;
        }
    }

    void train(const TemplateList& data)
    {
        (void) data;
    }

    void init()
    {
        if (targetName.isEmpty() || queryName.isEmpty() || outputString.isEmpty())
            return;

        FileList targetFiles = FileList::fromGallery(targetName);
        FileList queryFiles  = FileList::fromGallery(queryName);

        currentBlockRow = 0;
        currentBlockCol = 0;

        currentRow = 0;
        currentCol = 0;

        bufferedSize = 100;

        if (transposeMode)
        {
            // buffer 100 cols at a time
            fragmentsPerRow = bufferedSize;
            // a single col contains comparisons to all query files
            fragmentsPerCol = queryFiles.size();
        }
        else
        {
            // a single row contains comparisons to all target files
            fragmentsPerRow = targetFiles.size();
            // we output rows one at a time
            fragmentsPerCol = 1;
        }

        output = QSharedPointer<Output>(Output::make(outputString+"[targetGallery="+targetName+",queryGallery="+queryName+"]", targetFiles, queryFiles));
        output->blockRows = fragmentsPerCol;
        output->blockCols = fragmentsPerRow;
        output->initialize(targetFiles, queryFiles);

        output->setBlock(currentBlockRow, currentBlockCol);
    }

    QSharedPointer<Output> output;

    int bufferedSize;

    int currentRow;
    int currentCol;

    int currentBlockRow;
    int currentBlockCol;

    int fragmentsPerRow;
    int fragmentsPerCol;

public:
    OutputTransform() : TimeVaryingTransform(false,false) {}
};

BR_REGISTER(Transform, OutputTransform)

class FileExclusionTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(QString exclusionGallery READ get_exclusionGallery WRITE set_exclusionGallery RESET reset_exclusionGallery STORED false)
    BR_PROPERTY(QString, exclusionGallery, "")

    QSet<QString> excluded;

    void project(const Template &, Template &) const
    {
        qFatal("FileExclusion can't do anything here");
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach(const Template & srcTemp, src)
        {
            if (!excluded.contains(srcTemp.file))
                dst.append(srcTemp);
        }
    }

    void init()
    {
        if (exclusionGallery.isEmpty())
            return;
        FileList temp = FileList::fromGallery(exclusionGallery);
        excluded = QSet<QString>::fromList(temp.names());
    }
};

BR_REGISTER(Transform, FileExclusionTransform)

}

#include "misc.moc"
