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

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Treats each line as a file.
 * \author Josh Klontz \cite jklontz
 * \br_format Columns should be comma separated with first row containing headers.
 *            The first column in the file should be the path to the file to enroll.
 *            Other columns will be treated as file metadata.
 *
 * \br_related_plugin txtGallery
 */
class csvGallery : public FileGallery
{
    Q_OBJECT
    Q_PROPERTY(int fileIndex READ get_fileIndex WRITE set_fileIndex RESET reset_fileIndex)
    BR_PROPERTY(int, fileIndex, 0)

    FileList files;
    QStringList headers;

    ~csvGallery()
    {
        f.close();

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

        QMap<QString, int> columnCounts;

        { // Make header
            QStringList words;
            words.append("File");
            foreach (const QString &key, samples.keys()) {
                int count = 0;
                words.append(getCSVElement(key, samples[key], true, count));
                columnCounts.insert(key, count);
            }
            lines.append(words.join(","));
        }

        // Make table
        foreach (const File &file, files) {
            QStringList words;
            words.append(file.name);
            foreach (const QString &key, samples.keys()) {
                int count = columnCounts[key];
                words.append(getCSVElement(key, file.value(key), false, count));
            }
            lines.append(words.join(","));
        }

        QtUtils::writeFile(file, lines);
    }

    TemplateList readBlock(bool *done)
    {
        readOpen();
        *done = false;
        TemplateList templates;
        if (!file.exists()) {
            *done = true;
            return templates;
        }
        QRegExp regexp("\\s*,\\s*");

        if (f.pos() == 0)
        {
            // read a line
            QByteArray lineBytes = f.readLine();
            QString line = QString::fromLocal8Bit(lineBytes).trimmed();
            headers = line.split(regexp);
        }

        for (qint64 i = 0; i < this->readBlockSize && !f.atEnd(); i++){
            QByteArray lineBytes = f.readLine();
            QString line = QString::fromLocal8Bit(lineBytes).trimmed();

            QStringList words = line.split(regexp);
            if (words.size() != headers.size()) continue;
            File fi;
            for (int j=0; j<words.size(); j++) {
                if (j == 0) fi.name = words[j];
                else        fi.set(headers[j], words[j]);
            }
            templates.append(fi);
            templates.last().file.set("progress", f.pos());
        }
        *done = f.atEnd();

        return templates;
    }

    void write(const Template &t)
    {
        files.append(t.file);
    }

    static QString getCSVElement(const QString &key, const QVariant &value, bool header, int & columnCount)
    {
        if (header)
            columnCount = 1;

        if (value.canConvert<QString>()) {
            if (header) return key;
            else {
                if (columnCount != 1)
                    qFatal("Inconsistent datatype for key %s, csv file cannot be generated", qPrintable(key));
                return value.value<QString>();
            }
        } else if (value.canConvert<QPointF>()) {
            const QPointF point = value.value<QPointF>();
            if (header) {
                columnCount = 2;
                return key+"_X,"+key+"_Y";
            }
            else {
                if (columnCount != 2)
                    qFatal("Inconsistent datatype for key %s, csv file cannot be generated", qPrintable(key));

                return QString::number(point.x())+","+QString::number(point.y());
            }
        } else if (value.canConvert<QRectF>()) {
            const QRectF rect = value.value<QRectF>();
            if (header) {
                columnCount = 4;
                return key+"_X,"+key+"_Y,"+key+"_Width,"+key+"_Height";
            }
            else {
                if (columnCount != 4)
                    qFatal("Inconsistent datatype for key %s, csv file cannot be generated", qPrintable(key));

                return QString::number(rect.x())+","+QString::number(rect.y())+","+QString::number(rect.width())+","+QString::number(rect.height());
            }
        } else {
            if (header) return key;
            else {
                QString output = QString::number(std::numeric_limits<float>::quiet_NaN());
                for (int i = 1; i < columnCount; i++)
                    output += "," + QString::number(std::numeric_limits<float>::quiet_NaN());
                return output;
            }
        }
    }
};

BR_REGISTER(Gallery, csvGallery)

} // namespace br

#include "gallery/csv.moc"
