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

#include <QRegularExpression>
#include <QtConcurrentRun>
#ifndef BR_EMBEDDED
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QSqlRecord>
#endif // BR_EMBEDDED
#include <opencv2/highgui/highgui.hpp>
#include "openbr_internal.h"

#include "openbr/core/bee.h"
#include "openbr/core/common.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/qtutils.h"

#ifdef CVMATIO
#include "MatlabIO.hpp"
#include "MatlabIOContainer.hpp"
#endif

namespace br
{

/*!
 * \ingroup galleries
 * \brief Weka ARFF file format.
 * \author Josh Klontz \cite jklontz
 * http://weka.wikispaces.com/ARFF+%28stable+version%29
 */
class arffGallery : public Gallery
{
    Q_OBJECT
    QFile arffFile;

    TemplateList readBlock(bool *done)
    {
        (void) done;
        qFatal("Not implemented.");
        return TemplateList();
    }

    void write(const Template &t)
    {
        if (!arffFile.isOpen()) {
            arffFile.setFileName(file.name);
            arffFile.open(QFile::WriteOnly);
            arffFile.write("% OpenBR templates\n"
                           "@RELATION OpenBR\n"
                           "\n");

            const int dimensions = t.m().rows * t.m().cols;
            for (int i=0; i<dimensions; i++)
                arffFile.write(qPrintable("@ATTRIBUTE v" + QString::number(i) + " REAL\n"));
            arffFile.write(qPrintable("@ATTRIBUTE class string\n"));

            arffFile.write("\n@DATA\n");
        }

        arffFile.write(qPrintable(OpenCVUtils::matrixToStringList(t).join(',')));
        arffFile.write(qPrintable(",'" + t.file.get<QString>("Label") + "'\n"));
    }

    void init()
    {
        //
    }
};

BR_REGISTER(Gallery, arffGallery)

/*!
 * \ingroup galleries
 * \brief A binary gallery.
 *
 * Designed to be a literal translation of templates to disk.
 * Compatible with TemplateList::fromBuffer.
 * \author Josh Klontz \cite jklontz
 */
class galGallery : public Gallery
{
    Q_OBJECT
    QFile gallery;
    QDataStream stream;

    void init()
    {
        gallery.setFileName(file);
        if (file.get<bool>("remove"))
            gallery.remove();
        QtUtils::touchDir(gallery);
        QFile::OpenMode mode = QFile::ReadWrite;

        if (file.get<bool>("append"))
            mode |= QFile::Append;

        if (!gallery.open(mode))
            qFatal("Can't open gallery: %s", qPrintable(gallery.fileName()));
        stream.setDevice(&gallery);
    }

    TemplateList readBlock(bool *done)
    {
        if (stream.atEnd())
            gallery.seek(0);

        TemplateList templates;
        while ((templates.size() < readBlockSize) && !stream.atEnd()) {
            Template m;
            stream >> m;
            templates.append(m);
        }

        *done = stream.atEnd();
        return templates;
    }

    void write(const Template &t)
    {
        if (t.isEmpty() && t.file.isNull())
            return;

        stream << t;
    }
};

BR_REGISTER(Gallery, galGallery)

/*!
 * \ingroup galleries
 * \brief Reads/writes templates to/from folders.
 * \author Josh Klontz \cite jklontz
 * \param regexp An optional regular expression to match against the files extension.
 */
class EmptyGallery : public Gallery
{
    Q_OBJECT
    Q_PROPERTY(QString regexp READ get_regexp WRITE set_regexp RESET reset_regexp STORED false)
    BR_PROPERTY(QString, regexp, QString())

    void init()
    {
        QtUtils::touchDir(QDir(file.name));
    }

    TemplateList readBlock(bool *done)
    {
        TemplateList templates;
        *done = true;

        // Enrolling a null file is used as an idiom to initialize an algorithm
        if (file.isNull()) return templates;

        // Add immediate subfolders
        QDir dir(file);
        QList< QFuture<TemplateList> > futures;
        foreach (const QString &folder, QtUtils::naturalSort(dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot))) {
            const QDir subdir = dir.absoluteFilePath(folder);
            futures.append(QtConcurrent::run(&EmptyGallery::getTemplates, subdir));
        }
        foreach (const QFuture<TemplateList> &future, futures)
            templates.append(future.result());

        // Add root folder
        foreach (const QString &fileName, QtUtils::getFiles(file.name, false))
            templates.append(File(fileName, dir.dirName()));

        if (!regexp.isEmpty()) {
            QRegExp re(regexp);
            re.setPatternSyntax(QRegExp::Wildcard);
            for (int i=templates.size()-1; i>=0; i--) {
                if (!re.exactMatch(templates[i].file.fileName())) {
                    templates.removeAt(i);
                }
            }
        }

        return templates;
    }

    void write(const Template &t)
    {
        static QMutex diskLock;

        // Enrolling a null file is used as an idiom to initialize an algorithm
        if (file.name.isEmpty()) return;

        const QString newFormat = file.get<QString>("newFormat",QString());
        QString destination = file.name + "/" + (file.getBool("preservePath") ? t.file.path()+"/" : QString());
        destination += (newFormat.isEmpty() ? t.file.fileName() : t.file.baseName()+newFormat);

        QMutexLocker diskLocker(&diskLock); // Windows prefers to crash when writing to disk in parallel
        if (t.isNull()) {
            QtUtils::copyFile(t.file.resolved(), destination);
        } else {
            QScopedPointer<Format> format(Factory<Format>::make(destination));
            format->write(t);
        }
    }

    static TemplateList getTemplates(const QDir &dir)
    {
        const QStringList files = QtUtils::getFiles(dir, true);
        TemplateList templates; templates.reserve(files.size());
        foreach (const QString &file, files)
            templates.append(File(file, dir.dirName()));
        return templates;
    }
};

BR_REGISTER(Gallery, EmptyGallery)

/*!
 * \ingroup galleries
 * \brief Treats the gallery as a br::Format.
 * \author Josh Klontz \cite jklontz
 */
class DefaultGallery : public Gallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        *done = true;
        return TemplateList() << file;
    }

    void write(const Template &t)
    {
        QScopedPointer<Format> format(Factory<Format>::make(file));
        format->write(t);
    }
};

BR_REGISTER(Gallery, DefaultGallery)

/*!
 * \ingroup galleries
 * \brief Combine all templates into one large matrix and process it as a br::Format
 * \author Josh Klontz \cite jklontz
 */
class matrixGallery : public Gallery
{
    Q_OBJECT
    Q_PROPERTY(const QString extension READ get_extension WRITE set_extension RESET reset_extension STORED false)
    BR_PROPERTY(QString, extension, "mtx")

    TemplateList templates;

    ~matrixGallery()
    {
        if (templates.isEmpty())
            return;

        QScopedPointer<Format> format(Factory<Format>::make(getFormat()));
        format->write(Template(file, OpenCVUtils::toMat(templates.data())));
    }

    File getFormat() const
    {
        return file.name.left(file.name.size() - file.suffix().size()) + extension;
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        return TemplateList() << getFormat();
    }

    void write(const Template &t)
    {
        templates.append(t);
    }
};

BR_REGISTER(Gallery, matrixGallery)

/*!
 * \ingroup initializers
 * \brief Initialization support for memGallery.
 * \author Josh Klontz \cite jklontz
 */
class MemoryGalleries : public Initializer
{
    Q_OBJECT

    void initialize() const {}

    void finalize() const
    {
        galleries.clear();
    }

public:
    static QHash<File, TemplateList> galleries; /*!< TODO */
    static QHash<File, bool> aligned; /*!< TODO */
};

QHash<File, TemplateList> MemoryGalleries::galleries;
QHash<File, bool> MemoryGalleries::aligned;

BR_REGISTER(Initializer, MemoryGalleries)

/*!
 * \ingroup galleries
 * \brief A gallery held in memory.
 * \author Josh Klontz \cite jklontz
 */
class memGallery : public Gallery
{
    Q_OBJECT
    int block;

    void init()
    {
        block = 0;
        File galleryFile = file.name.mid(0, file.name.size()-4);
        if ((galleryFile.suffix() == "gal") && galleryFile.exists() && !MemoryGalleries::galleries.contains(file)) {
            QSharedPointer<Gallery> gallery(Factory<Gallery>::make(galleryFile));
            MemoryGalleries::galleries[file] = gallery->read();
            align(MemoryGalleries::galleries[file]);
            MemoryGalleries::aligned[file] = true;
        }
    }

    TemplateList readBlock(bool *done)
    {
        if (!MemoryGalleries::aligned[file]) {
            align(MemoryGalleries::galleries[file]);
            MemoryGalleries::aligned[file] = true;
        }

        TemplateList templates = MemoryGalleries::galleries[file].mid(block*readBlockSize, readBlockSize);
        *done = (templates.size() < readBlockSize);
        block = *done ? 0 : block+1;
        return templates;
    }

    void write(const Template &t)
    {
        MemoryGalleries::galleries[file].append(t);
        MemoryGalleries::aligned[file] = false;
    }

    static void align(TemplateList &templates)
    {
        if (!templates.empty() && templates[0].size() > 1) return;

        bool uniform = true;
        QVector<uchar> alignedData(templates.bytes<size_t>());
        size_t offset = 0;
        for (int i=0; i<templates.size(); i++) {
            Template &t = templates[i];
            if (t.size() > 1) qFatal("Can't handle multi-matrix template %s.", qPrintable(t.file.flat()));

            cv::Mat &m = t;
            if (m.data) {
                const size_t size = m.total() * m.elemSize();
                if (!m.isContinuous()) qFatal("Requires continuous matrix data of size %d for %s.", (int)size, qPrintable(t.file.flat()));
                memcpy(&(alignedData.data()[offset]), m.ptr(), size);
                m = cv::Mat(m.rows, m.cols, m.type(), &(alignedData.data()[offset]));
                offset += size;
            }
            uniform = uniform &&
                      (m.rows == templates.first().m().rows) &&
                      (m.cols == templates.first().m().cols) &&
                      (m.type() == templates.first().m().type());
        }

        templates.uniform = uniform;
        templates.alignedData = alignedData;
    }

};

BR_REGISTER(Gallery, memGallery)

/*!
 * \ingroup galleries
 * \brief Treats each line as a file.
 * \author Josh Klontz \cite jklontz
 *
 * Columns should be comma separated with first row containing headers.
 * The first column in the file should be the path to the file to enroll.
 * Other columns will be treated as file metadata.
 *
 * \see txtGallery
 */
class csvGallery : public Gallery
{
    Q_OBJECT
    Q_PROPERTY(int fileIndex READ get_fileIndex WRITE set_fileIndex RESET reset_fileIndex)
    BR_PROPERTY(int, fileIndex, 0)

    FileList files;

    ~csvGallery()
    {
        if (files.isEmpty()) return;

        QMap<QString,QVariant> samples;
        foreach (const File &file, files)
            foreach (const QString &key, file.localKeys())
                if (!samples.contains(key))
                    samples.insert(key, file.value(key));

        // Don't create columns in the CSV for these special fields
        samples.remove("Points");
        samples.remove("Rects");

        QStringList lines;
        lines.reserve(files.size()+1);

        { // Make header
            QStringList words;
            words.append("File");
            foreach (const QString &key, samples.keys())
                words.append(getCSVElement(key, samples[key], true));
            lines.append(words.join(","));
        }

        // Make table
        foreach (const File &file, files) {
            QStringList words;
            words.append(file.name);
            foreach (const QString &key, samples.keys())
                words.append(getCSVElement(key, file.value(key), false));
            lines.append(words.join(","));
        }

        QtUtils::writeFile(file, lines);
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        TemplateList templates;
        if (!file.exists()) return templates;

        QStringList lines = QtUtils::readLines(file);
        QRegExp regexp("\\s*,\\s*");
        QStringList headers;
        if (!lines.isEmpty()) headers = lines.takeFirst().split(regexp);

        foreach (const QString &line, lines) {
            QStringList words = line.split(regexp);
            if (words.size() != headers.size()) continue;
            File f;
            for (int i=0; i<words.size(); i++) {
                if (i == 0) f.name = words[i];
                else        f.set(headers[i], words[i]);
            }
            templates.append(f);
        }

        return templates;
    }

    void write(const Template &t)
    {
        files.append(t.file);
    }

    static QString getCSVElement(const QString &key, const QVariant &value, bool header)
    {
        if (value.canConvert<QString>()) {
            if (header) return key;
            else        return value.value<QString>();
        } else if (value.canConvert<QPointF>()) {
            const QPointF point = value.value<QPointF>();
            if (header) return key+"_X,"+key+"_Y";
            else        return QString::number(point.x())+","+QString::number(point.y());
        } else if (value.canConvert<QRectF>()) {
            const QRectF rect = value.value<QRectF>();
            if (header) return key+"_X,"+key+"_Y,"+key+"_Width,"+key+"_Height";
            else        return QString::number(rect.x())+","+QString::number(rect.y())+","+QString::number(rect.width())+","+QString::number(rect.height());
        } else {
            if (header) return key;
            else        return QString::number(std::numeric_limits<float>::quiet_NaN());
        }
    }
};

BR_REGISTER(Gallery, csvGallery)

/*!
 * \ingroup galleries
 * \brief Treats each line as a file.
 * \author Josh Klontz \cite jklontz
 *
 * The entire line is treated as the file path. An optional label may be specified using a space ' ' separator:
 *
\verbatim
<FILE>
<FILE>
...
<FILE>
\endverbatim
 * or
\verbatim
<FILE> <LABEL>
<FILE> <LABEL>
...
<FILE> <LABEL>
\endverbatim
 * \see csvGallery
 */
class txtGallery : public Gallery
{
    Q_OBJECT
    Q_PROPERTY(QString label READ get_label WRITE set_label RESET reset_label STORED false)
    BR_PROPERTY(QString, label, "")

    QStringList lines;

    ~txtGallery()
    {
        if (!lines.isEmpty())
            QtUtils::writeFile(file.name, lines);
    }

    TemplateList readBlock(bool *done)
    {
        TemplateList templates;
        foreach (const QString &line, QtUtils::readLines(file)) {
            int splitIndex = line.lastIndexOf(' ');
            if (splitIndex == -1) templates.append(File(line));
            else                  templates.append(File(line.mid(0, splitIndex), line.mid(splitIndex+1)));
        }
        *done = true;
        return templates;
    }

    void write(const Template &t)
    {
        QString line = t.file.name;
        if (!label.isEmpty())
            line += " " + t.file.get<QString>(label);
        lines.append(line);
    }
};

BR_REGISTER(Gallery, txtGallery)
/*!
 * \ingroup galleries
 * \brief Treats each line as a call to File::flat()
 * \author Josh Klontz \cite jklontz
 */
class flatGallery : public Gallery
{
    Q_OBJECT
    QStringList lines;

    ~flatGallery()
    {
        if (!lines.isEmpty())
            QtUtils::writeFile(file.name, lines);
    }

    TemplateList readBlock(bool *done)
    {
        TemplateList templates;
        foreach (const QString &line, QtUtils::readLines(file))
            templates.append(File(line));
        *done = true;
        return templates;
    }

    void write(const Template &t)
    {
        lines.append(t.file.flat());
    }
};

BR_REGISTER(Gallery, flatGallery)

/*!
 * \ingroup galleries
 * \brief A \ref sigset input.
 * \author Josh Klontz \cite jklontz
 */
class xmlGallery : public Gallery
{
    Q_OBJECT
    Q_PROPERTY(bool ignoreMetadata READ get_ignoreMetadata WRITE set_ignoreMetadata RESET reset_ignoreMetadata STORED false)
    BR_PROPERTY(bool, ignoreMetadata, false)
    FileList files;

    ~xmlGallery()
    {
        if (!files.isEmpty())
            BEE::writeSigset(file, files, ignoreMetadata);
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        return TemplateList(BEE::readSigset(file, ignoreMetadata));
    }

    void write(const Template &t)
    {
        files.append(t.file);
    }
};

BR_REGISTER(Gallery, xmlGallery)

/*!
 * \ingroup galleries
 * \brief Treat the file as a single binary template.
 * \author Josh Klontz \cite jklontz
 */
class templateGallery : public Gallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        *done = true;
        QByteArray data;
        QtUtils::readFile(file.name.left(file.name.size()-QString(".template").size()), data);
        return TemplateList() << Template(file, cv::Mat(1, data.size(), CV_8UC1, data.data()).clone());
    }

    void write(const Template &t)
    {
        (void) t;
        qFatal("No supported.");
    }

    void init()
    {
        //
    }
};

BR_REGISTER(Gallery, templateGallery)

/*!
 * \ingroup galleries
 * \brief Database input.
 * \author Josh Klontz \cite jklontz
 */
class dbGallery : public Gallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        TemplateList templates;
        br::File import = file.get<QString>("import", "");
        QString query = file.get<QString>("query");
        QString subset = file.get<QString>("subset", "");

#ifndef BR_EMBEDDED
        QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE");
        db.setDatabaseName(file);
        if (!db.open()) qFatal("Failed to open SQLite database %s.", qPrintable(file.name));

        if (!import.isNull()) {
            qDebug("Parsing %s", qPrintable(import.name));
            QStringList lines = QtUtils::readLines(import);
            QList<QStringList> cells; cells.reserve(lines.size());
            const QRegExp re("\\s*,\\s*");
            foreach (const QString &line, lines) {
                cells.append(line.split(re));
                if (cells.last().size() != cells.first().size()) qFatal("Column count mismatch.");
            }

            QStringList columns, qMarks;
            QList<QVariantList> variantLists;
            for (int i=0; i<cells[0].size(); i++) {
                bool isNumeric;
                cells[1][i].toInt(&isNumeric);
                columns.append(cells[0][i] + (isNumeric ? " INTEGER" : " STRING"));
                qMarks.append("?");

                QVariantList variantList; variantList.reserve(lines.size()-1);
                for (int j=1; j<lines.size(); j++) {
                    if (isNumeric) variantList << cells[j][i].toInt();
                    else           variantList << cells[j][i];
                }
                variantLists.append(variantList);
            }

            const QString &table = import.baseName();
            qDebug("Creating table %s", qPrintable(table));
            QSqlQuery q(db);
            if (!q.exec("CREATE TABLE " + table + " (" + columns.join(", ") + ");"))
                qFatal("%s.", qPrintable(q.lastError().text()));
            if (!q.prepare("insert into " + table + " values (" + qMarks.join(", ") + ")"))
                qFatal("%s.", qPrintable(q.lastError().text()));
            foreach (const QVariantList &vl, variantLists)
                q.addBindValue(vl);
            if (!q.execBatch()) qFatal("%s.", qPrintable(q.lastError().text()));
        }

        QSqlQuery q(db);
        if (query.startsWith('\'') && query.endsWith('\''))
            query = query.mid(1, query.size()-2);
        if (!q.exec(query))
            qFatal("%s.", qPrintable(q.lastError().text()));

        if ((q.record().count() == 0) || (q.record().count() > 3))
            qFatal("Query record expected one to three fields, got %d.", q.record().count());
        const bool hasMetadata = (q.record().count() >= 2);
        const bool hasFilter = (q.record().count() >= 3);

        QString labelName = "Label";
        if (q.record().count() >= 2)
            labelName = q.record().fieldName(1);

        // subset = seed:subjectMaxSize:numSubjects:subjectMinSize or
        // subset = seed:{Metadata,...,Metadata}:numSubjects
        int seed = 0, subjectMaxSize = std::numeric_limits<int>::max(), numSubjects = std::numeric_limits<int>::max(), subjectMinSize = 0;
        QList<QRegExp> metadataFields;
        if (!subset.isEmpty()) {
            const QStringList &words = subset.split(":");
            QtUtils::checkArgsSize("Input", words, 2, 4);
            if      (words[0] == "train") seed = 0;
            else if (words[0] == "test" ) seed = 1;
            else                          seed = QtUtils::toInt(words[0]);
            if (words[1].startsWith('{') && words[1].endsWith('}')) {
                foreach (const QString &regexp, words[1].mid(1, words[1].size()-2).split(","))
                    metadataFields.append(QRegExp(regexp));
                subjectMaxSize = metadataFields.size();
            } else {
                subjectMaxSize = QtUtils::toInt(words[1]);
            }
            numSubjects = words.size() >= 3 ? QtUtils::toInt(words[2]) : std::numeric_limits<int>::max();
            subjectMinSize = words.size() >= 4 ? QtUtils::toInt(words[3]) : subjectMaxSize;
        }

        srand(seed);

        typedef QPair<QString,QString> Entry; // QPair<File,Metadata>
        QHash<QString, QList<Entry> > entries; // QHash<Label, QList<Entry> >
        while (q.next()) {
            if (hasFilter && (seed >= 0) && (qHash(q.value(2).toString()) % 2 != (uint)seed % 2)) continue; // Ensures training and testing filters don't overlap

            if (metadataFields.isEmpty())
                entries[hasMetadata ? q.value(1).toString() : ""].append(QPair<QString,QString>(q.value(0).toString(), hasFilter ? q.value(2).toString() : ""));
            else
                entries[hasFilter ? q.value(2).toString() : ""].append(QPair<QString,QString>(q.value(0).toString(), hasMetadata ? q.value(1).toString() : ""));
        }

        QStringList labels = entries.keys();
        qSort(labels);

        if (hasFilter && ((labels.size() > numSubjects) || (numSubjects == std::numeric_limits<int>::max())))
            std::random_shuffle(labels.begin(), labels.end());

        foreach (const QString &label, labels) {
            QList<Entry> entryList = entries[label];
            if ((entryList.size() >= subjectMinSize) && (numSubjects > 0)) {

                if (!metadataFields.isEmpty()) {
                    QList<Entry> subEntryList;
                    foreach (const QRegExp &metadata, metadataFields) {
                        for (int i=0; i<entryList.size(); i++) {
                            if (metadata.exactMatch(entryList[i].second)) {
                                subEntryList.append(entryList.takeAt(i));
                                break;
                            }
                        }
                    }
                    if (subEntryList.size() == metadataFields.size())
                        entryList = subEntryList;
                    else
                        continue;
                }

                if (entryList.size() > subjectMaxSize)
                    std::random_shuffle(entryList.begin(), entryList.end());
                foreach (const Entry &entry, entryList.mid(0, subjectMaxSize)) {
                    templates.append(File(entry.first));
                    templates.last().file.set(labelName, label);
                }
                numSubjects--;
            }
        }

        db.close();
#endif // BR_EMBEDDED

        *done = true;
        return templates;
    }

    void write(const Template &t)
    {
        (void) t;
        qFatal("Not supported.");
    }

    void init()
    {
        //
    }
};

BR_REGISTER(Gallery, dbGallery)

/*!
 * \ingroup inputs
 * \brief Input from a google image search.
 * \author Josh Klontz \cite jklontz
 */
class googleGallery : public Gallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        TemplateList templates;

        static const QString search = "http://images.google.com/images?q=%1&start=%2";
        QString query = file.name.left(file.name.size()-7); // remove ".google"

#ifndef BR_EMBEDDED
        QNetworkAccessManager networkAccessManager;
        for (int i=0; i<100; i+=20) { // Retrieve 100 images
            QNetworkRequest request(search.arg(query, QString::number(i)));
            QNetworkReply *reply = networkAccessManager.get(request);

            while (!reply->isFinished())
                QThread::yieldCurrentThread();

            QString data(reply->readAll());
            delete reply;

            QStringList words = data.split("imgurl=");
            words.takeFirst(); // Remove header
            foreach (const QString &word, words) {
                QString url = word.left(word.indexOf("&amp"));
                url = url.replace("%2520","%20");
                int junk = url.indexOf('%', url.lastIndexOf('.'));
                if (junk != -1) url = url.left(junk);
                templates.append(File(url,query));
            }
        }
#endif // BR_EMBEDDED

        *done = true;
        return templates;
    }

    void write(const Template &)
    {
        qFatal("Not supported.");
    }
};

BR_REGISTER(Gallery, googleGallery)

/*!
 * \ingroup galleries
 * \brief Print template statistics.
 * \author Josh Klontz \cite jklontz
 */
class statGallery : public Gallery
{
    Q_OBJECT
    QSet<QString> subjects;
    QList<int> bytes;

    ~statGallery()
    {
        int emptyTemplates = 0;
        for (int i=bytes.size()-1; i>=0; i--)
            if (bytes[i] == 0) {
                bytes.removeAt(i);
                emptyTemplates++;
            }

        double bytesMean, bytesStdDev;
        Common::MeanStdDev(bytes, &bytesMean, &bytesStdDev);
        printf("Subjects: %d\nEmpty Templates: %d/%d\nBytes/Template: %.4g +/- %.4g\n",
               subjects.size(), emptyTemplates, emptyTemplates+bytes.size(), bytesMean, bytesStdDev);
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        return TemplateList() << file;
    }

    void write(const Template &t)
    {
        subjects.insert(t.file.get<QString>("Label"));
        bytes.append(t.bytes());
    }
};

BR_REGISTER(Gallery, statGallery)

/*!
 * \ingroup galleries
 * \brief Implements the FDDB detection format.
 * \author Josh Klontz \cite jklontz
 *
 * http://vis-www.cs.umass.edu/fddb/README.txt
 */
class FDDBGallery : public Gallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        *done = true;
        QStringList lines = QtUtils::readLines(file);
        TemplateList templates;
        while (!lines.empty()) {
            const QString fileName = lines.takeFirst();
            int numDetects = lines.takeFirst().toInt();
            for (int i=0; i<numDetects; i++) {
                const QStringList detect = lines.takeFirst().split(' ');
                Template t(fileName);
                QList<QVariant> faceList; //to be consistent with slidingWindow
                if (detect.size() == 5) { //rectangle
                    faceList.append(QRectF(detect[0].toFloat(), detect[1].toFloat(), detect[2].toFloat(), detect[3].toFloat()));
                    t.file.set("Confidence", detect[4].toFloat());
                } else if (detect.size() == 6) { //ellipse
                    float x = detect[3].toFloat(),  
                          y = detect[4].toFloat(),
                          radius = detect[1].toFloat();
                    faceList.append(QRectF(x - radius,y - radius,radius * 2.0, radius * 2.0));
                    t.file.set("Confidence", detect[5].toFloat());
                } else {
                    qFatal("Unknown FDDB annotation format.");
                }
                t.file.set("Face", faceList);
                t.file.set("Label",QString("face"));
                templates.append(t);
            }
        }
        return templates;
    }

    void write(const Template &t)
    {
        (void) t;
        qFatal("Not implemented.");
    }

    void init()
    {
        //
    }
};

BR_REGISTER(Gallery, FDDBGallery)

/*!
 * \ingroup galleries
 * \brief Text format for associating anonymous landmarks with images.
 * \author Josh Klontz \cite jklontz
 *
 * \code
 * file_name:x1,y1,x2,y2,...,xn,yn
 * file_name:x1,y1,x2,y2,...,xn,yn
 * ...
 * file_name:x1,y1,x2,y2,...,xn,yn
 * \endcode
 */
class landmarksGallery : public Gallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        *done = true;
        TemplateList templates;
        foreach (const QString &line, QtUtils::readLines(file)) {
            const QStringList words = line.split(':');
            if (words.size() != 2) qFatal("Expected exactly one ':' in: %s.", qPrintable(line));
            File file(words[0]);
            const QList<float> vals = QtUtils::toFloats(words[1].split(','));
            if (vals.size() % 2 != 0) qFatal("Expected an even number of comma-separated values.");
            QList<QPointF> points; points.reserve(vals.size()/2);
            for (int i=0; i<vals.size(); i+=2)
                points.append(QPointF(vals[i], vals[i+1]));
            file.setPoints(points);
            templates.append(file);
        }
        return templates;
    }

    void write(const Template &t)
    {
        (void) t;
        qFatal("Not implemented.");
    }

    void init()
    {
        //
    }
};

BR_REGISTER(Gallery, landmarksGallery)

/*!
 * \ingroup galleries
 * \brief Prints top attributes.
 * \author Babatunde Ogunfemi \cite baba1472
 *
 * Prints to std out the top predictions. An optional threshold may be specified using a space ' ' separator:
 *
 */
class topPredictionsGallery : public Gallery
{
    Q_OBJECT
    Q_PROPERTY(float out_count READ get_out_count WRITE set_out_count RESET reset_out_count STORED false)
    BR_PROPERTY(float, out_count, 3)

    QList<QStringList> attributes;

    ~topPredictionsGallery()
    {
        QString result = QString("\n====================\n  # of Templates: %1 \n====================").arg(QString::number(attributes.count()));
        QtUtils::writeFile(file.name, result);
        if (!attributes.isEmpty())
        {
            foreach (const QStringList &a_list, attributes)
            {
                QString result = QString("\n\n+++++++++++++++++++++++++++++++++++++++++++++");
                QtUtils::writeFile(file.name, result);

                foreach (const QString &output, a_list)
                    QtUtils::writeFile(file.name, output);

                result = QString("+++++++++++++++++++++++++++++++++++++++++++++");
                QtUtils::writeFile(file.name, result);
            }
        }
    }

    //Pure virtial function so needs an implementation
    TemplateList readBlock(bool *done)
    {
        (void) done;
        qFatal("Unsupported.");
        return TemplateList();
    }

    void write(const Template &t)
    {
        QList<QString> keys = t.file.localKeys();
        QMap <float, QString> data;
        QStringList top_attributes;
        foreach (const QString &key, keys)
        {
            if (key.startsWith("predicted_"))
            {
                float val = t.file.get<float>(key);
                data.insert(val, key); //use float as key to keep in order
            }
        }

        QMapIterator <float, QString> it(data);
        QString result;
        it.toBack();
        if (out_count > 1)
        {
            if (data.count() < out_count)
                out_count = data.count();

            for (int i = 0; i < out_count; i++)
            {
                it.previous();
                result = QString("%1)	%2 : %3").arg(QString::number(i + 1), it.value(), QString::number((float)it.key()));
                top_attributes.append(result);
            }
        }
        else
        {
            int count = 0;
            it.toBack();
            while (it.hasPrevious())
            {
                it.previous();
                if (it.key() >= out_count)
                {
                    result = QString("%1)	%2 : %3").arg(QString::number(count + 1), it.value(), QString::number((float)it.key()));
                    top_attributes.append(result);
                    count++;
                }
            }
        }
        attributes.append(top_attributes);
    }
};

BR_REGISTER(Gallery, topPredictionsGallery)

#ifdef CVMATIO

using namespace cv;

class vbbGallery : public Gallery
{
    Q_OBJECT

    void init()
    {
        MatlabIO matio;
        QString filename = (Globals->path.isEmpty() ? "" : Globals->path + "/") + file.name;
        bool ok = matio.open(filename.toStdString(), "r");
        if (!ok) qFatal("Couldn't open the vbb file");

        vector<MatlabIOContainer> variables;
        variables = matio.read();
        matio.close();

        double vers = variables[1].data<Mat>().at<double>(0,0);
        if (vers != 1.4) qFatal("This is an old vbb version, we don't mess with that.");

        A = variables[0].data<vector<vector<MatlabIOContainer> > >().at(0);
        objLists = A.at(1).data<vector<MatlabIOContainer> >();

        // start at the first frame (duh!)
        currFrame = 0;
    }

    TemplateList readBlock(bool *done)
    {
        *done = false;
        Template rects(file);
        if (objLists[currFrame].typeEquals<vector<vector<MatlabIOContainer> > >()) {
            vector<vector<MatlabIOContainer> > bbs = objLists[currFrame].data<vector<vector<MatlabIOContainer> > >();
            for (unsigned int i=0; i<bbs.size(); i++) {
                vector<MatlabIOContainer> bb = bbs[i];
                Mat pos = bb[1].data<Mat>();
                double left = pos.at<double>(0,0);
                double top = pos.at<double>(0,1);
                double width = pos.at<double>(0,2);
                double height = pos.at<double>(0,3);
                rects.file.appendRect(QRectF(left, top, width, height));
            }
        }
        TemplateList tl;
        tl.append(rects);
        if (++currFrame == (int)objLists.size()) *done = true;
        return tl;
    }

    void write(const Template &t)
    {
        (void)t; qFatal("Not implemented");
    }

private:
    // this holds a bunch of stuff, maybe we'll use it all later
    vector<MatlabIOContainer> A;
    // this, a field in A, holds bounding boxes for each frame
    vector<MatlabIOContainer> objLists;
    int currFrame;
};

BR_REGISTER(Gallery, vbbGallery)

#endif

} // namespace br

#include "gallery.moc"
