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
 * \ingroup transforms
 * \brief Normalize points to be relative to a single point
 * \author Scott Klum \cite sklum
 */
class PointDisplacementTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        QList<QPointF> points = dst.points();
        QList<QPointF> normalizedPoints;

        for (int i=0; i<points.size(); i++)
            for (int j=i+1; j<points.size(); j++) {
                    QPointF normalizedPoint = points[i]-points[j];
                    normalizedPoint.setX(pow(normalizedPoint.x(),2));
                    normalizedPoint.setY(pow(normalizedPoint.y(),2));
                    normalizedPoints.append(normalizedPoint);
                }

        dst.setPoints(normalizedPoints);
    }
};

BR_REGISTER(Transform, PointDisplacementTransform)

} // namespace br

#include "metadata/pointdisplacement.moc"
