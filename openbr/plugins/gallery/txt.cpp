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

namespace br
{

/*!
 * \ingroup galleries
 * \brief Treats each line as a file.
 * \author Josh Klontz \cite jklontz
 * \br_format The entire line is treated as the file path.
 *
 * <FILE>
 * <FILE>
 * ...
 * <FILE>
 *
 * An optional label may be specified using a space ' ' separator:
 *
 * <FILE> <LABEL>
 * <FILE> <LABEL>
 * ...
 * <FILE> <LABEL>
 *
 * \br_related_plugin csvGallery
 */
class txtGallery : public FileGallery
{
    Q_OBJECT
    Q_PROPERTY(QString label READ get_label WRITE set_label RESET reset_label STORED false)
    BR_PROPERTY(QString, label, "")

    TemplateList readBlock(bool *done)
    {
        readOpen();
        *done = false;
        if (f.atEnd())
            f.seek(0);

        TemplateList templates;

        for (qint64 i = 0; i < readBlockSize; i++)
        {
            QByteArray lineBytes = f.readLine();
            QString line = QString::fromLocal8Bit(lineBytes).trimmed();

            if (!line.isEmpty()){
                int splitIndex = line.lastIndexOf(' ');
                if (splitIndex == -1) templates.append(File(line));
                else                  templates.append(File(line.mid(0, splitIndex), line.mid(splitIndex+1)));
                templates.last().file.set("progress", this->position());
            }

            if (f.atEnd()) {
                *done=true;
                break;
            }
        }

        return templates;
    }

    void write(const Template &t)
    {
        writeOpen();
        QString line = t.file.name;
        if (!label.isEmpty())
            line += " " + t.file.get<QString>(label);

        f.write((line+"\n").toLocal8Bit() );
    }
};

BR_REGISTER(Gallery, txtGallery)

} // namespace br

#include "gallery/txt.moc"
