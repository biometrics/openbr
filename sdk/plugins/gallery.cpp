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

#ifndef BR_EMBEDDED
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QSqlRecord>
#endif // BR_EMBEDDED
#include <openbr_plugin.h>

#include "core/bee.h"
#include "core/opencvutils.h"
#include "core/qtutils.h"

namespace br
{

QDataStream &operator<<(QDataStream &stream, const Template &t)
{
    return stream << static_cast<const QList<cv::Mat>&>(t) << t.file;
}

QDataStream &operator>>(QDataStream &stream, Template &t)
{
    return stream >> static_cast<QList<cv::Mat>&>(t) >> t.file;
}

/*!
 * \ingroup galleries
 * \brief A binary gallery.
 * \author Josh Klontz \cite jklontz
 */
class galGallery : public Gallery
{
    Q_OBJECT
    QFile gallery;
    QDataStream stream;

    ~galGallery()
    {
        gallery.close();
    }

    void init()
    {
        gallery.setFileName(file);
        QtUtils::touchDir(gallery);

        if (file.contains("append")) {
            if (!gallery.open(QFile::Append))
                qFatal("Can't open [%s] for appending.", qPrintable(gallery.fileName()));
        } else if (gallery.exists()) {
            if (!gallery.open(QFile::ReadOnly))
                qFatal("Can't open [%s] for reading.", qPrintable(gallery.fileName()));
        } else {
            if (!gallery.open(QFile::WriteOnly))
                qFatal("Can't open [%s] for writing.", qPrintable(gallery.fileName()));
        }

        stream.setDevice(&gallery);
    }

    bool isUniversal() const
    {
        return true;
    }

    TemplateList readBlock(bool *done)
    {
        *done = false;

        TemplateList templates;
        while (templates.size() < Globals->blockSize) {
            if (stream.atEnd()) {
                *done = true;
                gallery.seek(0);
                break;
            }

            Template m;
            stream >> m;
            templates.append(m);
        }
        return templates;
    }

    void write(const Template &t)
    {
        stream << t;
    }
};

BR_REGISTER(Gallery, galGallery)

/*!
 * \ingroup galleries
 * \brief Reads and writes templates to folders of images.
 * \author Josh Klontz \cite jklontz
 */
class EmptyGallery : public Gallery
{
    Q_OBJECT
    Q_PROPERTY(bool cache READ get_cache WRITE set_cache RESET reset_cache)
    Q_PROPERTY(QString postfix READ get_postfix WRITE set_postfix RESET reset_postfix)
    BR_PROPERTY(bool, cache, false)
    BR_PROPERTY(QString, postfix, "")

    void init()
    {
        QtUtils::touchDir(QDir(file.name));
    }

    bool isUniversal() const
    {
        return false;
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;

        // Enrolling a null file is used as an idiom to initialize an algorithm
        if (file.name.isEmpty()) return TemplateList();

        TemplateList templates;

        // Add immediate subfolders
        QDir dir(file);
        foreach (const QString &folder, dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot))
            foreach (const QString &file, QtUtils::getFiles(dir.absoluteFilePath(folder), true))
                templates.append(File(file, folder));

        // Add root folder
        foreach (const QString &fileName, QtUtils::getFiles(file.name, false))
            templates.append(File(fileName, dir.dirName()));

        return templates;
    }

    void write(const Template &t)
    {
        static QMutex diskLock;

        // Enrolling a null file is used as an idiom to initialize an algorithm
        if (file.name.isEmpty()) return;

        QMutexLocker diskLocker(&diskLock);
        const QString destination = file.name + "/" + t.file.baseName() + postfix + ".png";
        if (!cache || !QFileInfo(destination).exists()) {
            if (t.isNull()) QFile::copy(t.file.name, destination);
            else            OpenCVUtils::saveImage(t, destination);
        }
    }
};

BR_REGISTER(Gallery, EmptyGallery)

/*!
 * \ingroup galleries
 * \brief Treats the gallery as a single image.
 * \author Josh Klontz \cite jklontz
 */
class DefaultGallery : public Gallery
{
    Q_OBJECT

    bool isUniversal() const
    {
        return false;
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        return TemplateList() << file;
    }

    void write(const Template &t)
    {
        QtUtils::touchDir(QFileInfo(file.name));
        OpenCVUtils::saveImage(t, file.name);
    }
};

BR_REGISTER(Gallery, DefaultGallery)

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

    bool isUniversal() const
    {
        return true;
    }

    TemplateList readBlock(bool *done)
    {
        if (!MemoryGalleries::aligned[file]) {
            align(MemoryGalleries::galleries[file]);
            MemoryGalleries::aligned[file] = true;
        }

        TemplateList templates = MemoryGalleries::galleries[file].mid(block*Globals->blockSize, Globals->blockSize);
        *done = (templates.size() < Globals->blockSize);
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
        bool uniform = true;
        QVector<uchar> alignedData(templates.bytes<size_t>());
        size_t offset = 0;
        for (int i=0; i<templates.size(); i++) {
            Template &t = templates[i];
            if (t.size() > 1) qFatal("memGallery::align can't handle multi-matrix template %s.", qPrintable(t.file.flat()));

            cv::Mat &m = t;
            if (m.data) {
                const size_t size = m.total() * m.elemSize();
                if (!m.isContinuous()) qFatal("memGallery::align requires continuous matrix data of size %d for %s.", (int)size, qPrintable(t.file.flat()));
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

    bool isUniversal() const
    {
        return false;
    }

    ~csvGallery()
    {
        if (files.isEmpty()) return;

        QStringList keys;
        foreach (const File &file, files)
            foreach (const QString &key, file.localKeys())
                if (!keys.contains(key)) keys += key;
        qSort(keys);

        const int rows = files.size();
        const int columns = keys.size();
        QSharedPointer<Output> output(Output::make(file, keys, files));

        for (int i=0; i<rows; i++)
            for (int j=0; j<columns; j++)
                if (keys[j] == "Label") output->setRelative(files[i].label(), i, j);
                else                    output->setRelative(files[i].getFloat(keys[j], std::numeric_limits<float>::quiet_NaN()), i, j);
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        TemplateList templates;
        if (!file.exists()) return templates;

        QStringList lines = QtUtils::readLines(file);
        if (!lines.isEmpty()) lines.removeFirst(); // Remove header

        foreach (const QString &line, lines) {
            QStringList words = line.split(',');
            if (words.isEmpty()) continue;
            templates.append(File(words[fileIndex], words.size() > 1 ? words.takeLast() : ""));
        }

        return templates;
    }

    void write(const Template &t)
    {
        files.append(t.file);
    }
};

BR_REGISTER(Gallery, csvGallery)

/*!
 * \ingroup galleries
 * \brief Treats each line as a file.
 * \author Josh Klontz \cite jklontz
 *
 * The entire line is treated as the file path.
 *
 * \see csvGallery
 */
class txtGallery : public Gallery
{
    Q_OBJECT

    QStringList lines;

    ~txtGallery()
    {
        if (!lines.isEmpty()) QtUtils::writeFile(file.name, lines);
    }

    bool isUniversal() const
    {
        return false;
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        TemplateList templates;
        if (!file.exists()) return templates;

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

BR_REGISTER(Gallery, txtGallery)

/*!
 * \ingroup galleries
 * \brief A \ref sigset input.
 * \author Josh Klontz \cite jklontz
 */
class xmlGallery : public Gallery
{
    Q_OBJECT
    Q_PROPERTY(bool ignoreMetadata READ get_ignoreMetadata WRITE set_ignoreMetadata RESET reset_ignoreMetadata)
    BR_PROPERTY(bool, ignoreMetadata, false)

    bool isUniversal() const
    {
        return false;
    }

    TemplateList readBlock(bool *done)
    {
        TemplateList templates;
        foreach (const File &signature, BEE::readSigset(file.name, ignoreMetadata))
            templates.append(signature);
        *done = true;
        return templates;
    }

    void write(const Template &t)
    {
        (void) t;
        qFatal("Writing to an xmlGallery not supported.");
    }
};

BR_REGISTER(Gallery, xmlGallery)

/*!
 * \ingroup galleries
 * \brief Database input.
 * \author Josh Klontz \cite jklontz
 */
class dbGallery : public Gallery
{
    Q_OBJECT

    bool isUniversal() const
    {
        return false;
    }

    TemplateList readBlock(bool *done)
    {
        TemplateList templates;
        br::File import = file.getString("import", "");
        QString query = file.getString("query");
        QString subset = file.getString("subset", "");

#ifndef BR_EMBEDDED
        static QSqlDatabase db;
        if (!db.isValid()) db = QSqlDatabase::addDatabase("QSQLITE");
        db.setDatabaseName(file);
        if (!db.open()) qFatal("Input::loadDatabase failed to open SQLite database %s.", qPrintable((QString)file));

        if (!import.isNull()) {
            qDebug("Parsing %s", qPrintable((QString)import));
            QStringList lines = QtUtils::readLines(import);
            QList<QStringList> cells; cells.reserve(lines.size());
            const QRegExp re("\\s*,\\s*");
            foreach (const QString &line, lines)
                cells.append(line.split(re));
            if (cells.size() < 2) qFatal("Input::loadDatabase expected at least two rows in %s.", qPrintable((QString)import));
            if (cells[0].size() != cells[1].size()) qFatal("Input::loadDatabase column count mismatch.");

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
                qFatal("Input::loadDatabase %s.", qPrintable(q.lastError().text()));
            if (!q.prepare("insert into " + table + " values (" + qMarks.join(", ") + ")"))
                qFatal("Input::loadDatabase %s.", qPrintable(q.lastError().text()));
            foreach (const QVariantList &vl, variantLists)
                q.addBindValue(vl);
            if (!q.execBatch()) qFatal("Input::loadDatabase %s.", qPrintable(q.lastError().text()));
        }

        QSqlQuery q(db);
        if (query.startsWith('\'') && query.endsWith('\''))
            query = query.mid(1, query.size()-2);
        if (!q.exec(query))
            qFatal("Input::loadDatabase %s.", qPrintable(q.lastError().text()));
        if ((q.record().count() == 0) || (q.record().count() > 3))
            qFatal("Input::loadDatabase query record expected one to three fields, got %d.", q.record().count());
        const bool hasMetadata = (q.record().count() >= 2);
        const bool hasFilter = (q.record().count() >= 3);

        // subset = seed:subjectMaxSize:numSubjects:subjectMinSize or
        // subset = seed:{Metadata,...,Metadata}:numSubjects
        int seed = 0, subjectMaxSize = std::numeric_limits<int>::max(), numSubjects = std::numeric_limits<int>::max(), subjectMinSize = 0;
        QList<QRegExp> metadataFields;
        if (!subset.isEmpty()) {
            const QStringList &words = subset.split(":");
            QtUtils::checkArgsSize("Input", words, 2, 4);
            seed = QtUtils::toInt(words[0]);
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
        if (hasFilter && ((labels.size() > numSubjects) || (numSubjects == std::numeric_limits<int>::max())))
            std::random_shuffle(labels.begin(), labels.end());
        else
            qSort(labels);

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
                foreach (const Entry &entry, entryList.mid(0, subjectMaxSize))
                    templates.append(File(entry.first, label));
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
        qFatal("Writing to a dbGallery not supported.");
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

    bool isUniversal() const
    {
        return false;
    }

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
                QCoreApplication::processEvents();

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

    void write(const Template &t)
    {
        (void) t;
        qFatal("Writing to a txtGallery not supported.");
    }
};

BR_REGISTER(Gallery, googleGallery)

} // namespace br

#include "gallery.moc"
