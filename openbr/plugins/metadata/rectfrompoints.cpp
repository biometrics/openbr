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
 * \brief Create rect from landmarks.
 * \author Scott Klum \cite sklum
 * \todo Padding should be a percent of total image size
 */

class RectFromPointsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    Q_PROPERTY(double padding READ get_padding WRITE set_padding RESET reset_padding STORED false)
    Q_PROPERTY(bool useAspectRatio READ get_useAspectRatio WRITE set_useAspectRatio RESET reset_useAspectRatio STORED false)
    Q_PROPERTY(double aspectRatio READ get_aspectRatio WRITE set_aspectRatio RESET reset_aspectRatio STORED false)
    BR_PROPERTY(QList<int>, indices, QList<int>())
    BR_PROPERTY(double, padding, 0)
    BR_PROPERTY(bool, useAspectRatio, true)
    BR_PROPERTY(double, aspectRatio, 1.0)

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        if (src.file.points().isEmpty()) {
            if (Globals->verbose) qWarning("No landmarks");
            return;
        }

        int minX, minY;
        minX = minY = std::numeric_limits<int>::max();
        int maxX, maxY;
        maxX = maxY = -std::numeric_limits<int>::max();

        QList<QPointF> points;

        int numPoints = (indices.empty() ? src.file.points().size() : indices.size());
        for (int idx = 0; idx < numPoints; idx++) {
            int index = indices.empty() ? idx : indices[idx];
            if (src.file.points().size() > index) {
                if (src.file.points()[index].x() <= 0 ||
                    src.file.points()[index].y() <= 0)   continue;
                if (src.file.points()[index].x() < minX) minX = src.file.points()[index].x();
                if (src.file.points()[index].x() > maxX) maxX = src.file.points()[index].x();
                if (src.file.points()[index].y() < minY) minY = src.file.points()[index].y();
                if (src.file.points()[index].y() > maxY) maxY = src.file.points()[index].y();
                points.append(src.file.points()[index]);
            }
            else qFatal("Incorrect indices");
        }

        double width = maxX-minX;
        double deltaWidth = width*padding;
        width += deltaWidth;

        double height = maxY-minY;
        double deltaHeight = height*padding;
        if (useAspectRatio)
            deltaHeight = width/aspectRatio - height;
        height += deltaHeight;

        const int x = std::max(0.0, minX - deltaWidth/2.0);
        const int y = std::max(0.0, minY - deltaHeight/2.0);

        dst.file.setPoints(points);
        dst.file.appendRect(QRectF(x, y, std::min((double)src.m().cols-x, width), std::min((double)src.m().rows-y, height)));
    }
};

BR_REGISTER(Transform, RectFromPointsTransform)

} // namespace br

#include "metadata/rectfrompoints.moc"
