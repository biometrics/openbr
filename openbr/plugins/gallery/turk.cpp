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
#include <openbr/core/common.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief For Amazon Mechanical Turk datasets
 * \author Scott Klum \cite sklum
 */
class turkGallery : public Gallery
{
    Q_OBJECT

    struct Attribute : public QStringList
    {
        QString name;
        Attribute(const QString &str = QString())
        {
            const int i = str.indexOf('[');
            name = str.mid(0, i);
            if (i != -1)
                append(str.mid(i+1, str.length()-i-2).split(","));
        }

        Attribute normalized() const
        {
            bool ok;
            QList<float> values;
            foreach (const QString &value, *this) {
                values.append(value.toFloat(&ok));
                if (!ok)
                    qFatal("Can't normalize non-numeric vector!");
            }

            Attribute normal(name);
            float sum = Common::Sum(values);
            if (sum == 0) sum = 1;
            for (int i=0; i<values.size(); i++)
                normal.append(QString::number(values[i] / sum));
            return normal;
        }
    };

    TemplateList readBlock(bool *done)
    {
        *done = true;
        QStringList lines = QtUtils::readLines(file);
        QList<Attribute> headers;
        if (!lines.isEmpty())
            foreach (const QString &header, parse(lines.takeFirst()))
                headers.append(header);

        TemplateList templates;
        foreach (const QString &line, lines) {
            QStringList words = parse(line);
            if (words.size() != headers.size())
                qFatal("turkGallery invalid column count");

            File f;
            f.name = words[0];
            f.set("Label", words[0].mid(0,5));

            for (int i=1; i<words.size(); i++) {
                Attribute ratings = Attribute(words[i]).normalized();
                if (headers[i].size() != ratings.size())
                    qFatal("turkGallery invalid attribute count");
                for (int j=0; j<ratings.size(); j++)
                    f.set(headers[i].name + "_" + headers[i][j], ratings[j]);
            }
            templates.append(f);
        }

        return templates;
    }

    void write(const Template &)
    {
        qFatal("turkGallery write not implemented.");
    }
};

BR_REGISTER(Gallery, turkGallery)

} // namespace br

#include "gallery/turk.moc"
