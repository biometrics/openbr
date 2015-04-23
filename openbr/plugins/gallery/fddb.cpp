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
 * \brief Implements the FDDB detection format.
 * \author Josh Klontz \cite jklontz
 *
 * \br_link http://vis-www.cs.umass.edu/fddb/README.txt
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

} // namespace br

#include "gallery/fddb.moc"
