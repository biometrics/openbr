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
 * \brief Text format for associating anonymous landmarks with images.
 * \author Josh Klontz \cite jklontz
 * \br_format The input should be formatted as follows:
 *
 * file_name:x1,y1,x2,y2,...,xn,yn
 * file_name:x1,y1,x2,y2,...,xn,yn
 * ...
 * file_name:x1,y1,x2,y2,...,xn,yn
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

} // namespace br

#include "gallery/landmarks.moc"
