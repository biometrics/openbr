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

namespace br
{

/*!
 * \ingroup galleries
 * \brief Print Template statistics.
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

} // namespace br

#include "gallery/stat.moc"
