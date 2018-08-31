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

    FileList files;
    QStringList headers;

    ~csvGallery()
    {
        f.close();
        if (files.isEmpty()) return;

        QSet<QString> samples;
        foreach (const File &file, files)
            foreach (const QString &key, file.localKeys())
                samples.insert(key);

        QStringList lines;
        lines.reserve(files.size()+1);

        // Make header
        QStringList keys = samples.values();
        keys.sort();
        lines.append(QStringList(QStringList("File") + keys).join(","));

        // Make table
        foreach (const File &file, files) {
            QStringList words;
            words.append(file.name);
            foreach (const QString &key, keys) {
                QString value = QtUtils::toString(file.value(key));
                if (value.contains(","))
                    value = '"' + value + '"';
                words.append(value);
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

        if (f.pos() == 0) {
            // read header
            QByteArray lineBytes = f.readLine();
            QString line = QString::fromLocal8Bit(lineBytes).trimmed();
            QRegExp regexp("\\s*,\\s*");
            headers = line.split(regexp);
        }

        for (qint64 i = 0; i < this->readBlockSize && !f.atEnd(); i++) {
            const QVariantList values = parseLine(f.readLine());
            if (values.size() != headers.size()) continue;

            File in;
            for (int j=0; j<values.size(); j++) {
                if (j == 0) in.name = values[j].toString();
                else        in.set(headers[j], values[j].toString());
            }
            in.set("progress", f.pos());
            templates.append(in);
        }
        *done = f.atEnd();
        return templates;
    }

    void write(const Template &t)
    {
        files.append(t.file);
    }

    static QVariantList parseLine(const QByteArray bytes)
    {
        bool inQuote(false);
        QVariantList values;
        QString value = QString();
        const QString line = QString::fromLocal8Bit(bytes).trimmed();
        for (int i=0; i<line.size(); i++) {
            const QChar c = line[i];
            if (c == '"') {
                inQuote = !inQuote;
                continue;
            } else if (c == ',' && !inQuote) {
                values.append(QVariant(value));
                value = QString();
            } else
                value.append(c);
        }
        if (!value.isEmpty())
            values.append(QVariant(value));

        return values;
    }
};

BR_REGISTER(Gallery, csvGallery)

} // namespace br

#include "gallery/csv.moc"
