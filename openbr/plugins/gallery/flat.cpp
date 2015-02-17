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
 * \brief Treats each line as a call to File::flat()
 * \author Josh Klontz \cite jklontz
 */
class flatGallery : public FileGallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        readOpen();
        *done = false;
        if (f.atEnd())
            f.seek(0);

        TemplateList templates;

        for (qint64 i = 0; i < readBlockSize; i++)
        {
            QByteArray line = f.readLine();

            if (!line.isEmpty()) {
                templates.append(File(QString::fromLocal8Bit(line).trimmed()));
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
        f.write((t.file.flat()+"\n").toLocal8Bit() );
    }
};

BR_REGISTER(Gallery, flatGallery)

} // namespace br

#include "gallery/flat.moc"
