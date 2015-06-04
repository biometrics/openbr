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
 * \brief Set points relative to a rect
 * \author Jordan Cheney \cite JordanCheney
 */
class SetPointsInRectTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    void projectMetadata(const File &src, File &dst) const
    {
        dst  = src;

        QList<QRectF> rects = src.rects();
        if (rects.size() != 1)
            qFatal("Must have one and only one rect per template");
        QRectF rect = rects.first();

        QList<QPointF> srcPoints = src.points();
        QList<QPointF> dstPoints;
        foreach (const QPointF &point, srcPoints)
            dstPoints.append(QPointF(point.x() - rect.x(), point.y() - rect.y()));

        dst.setPoints(dstPoints);
    }
};

BR_REGISTER(Transform, SetPointsInRectTransform)

} // namespace br

#include "metadata/setpointsinrect.moc"
