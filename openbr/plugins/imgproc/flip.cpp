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
 * \brief Flips the image about an axis.
 * \author Josh Klontz \cite jklontz
 */
class FlipTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Axis)
    Q_PROPERTY(Axis axis READ get_axis WRITE set_axis RESET reset_axis STORED false)

public:
    /*!< */
    enum Axis { X = 0,
                Y = 1,
                Both = -1 };

private:
    BR_PROPERTY(Axis, axis, Y)

    void project(const Template &src, Template &dst) const
    {
        cv::flip(src, dst, axis);

        QList<QPointF> flippedPoints;
        foreach(const QPointF &point, src.file.points()) {
            // Check for missing data using the QPointF(-1,-1) convention
            if (point != QPointF(-1,-1)) {
                if (axis == Y) {
                    flippedPoints.append(QPointF(src.m().cols-point.x(),point.y()));
                } else if (axis == X) {
                    flippedPoints.append(QPointF(point.x(),src.m().rows-point.y()));
                } else {
                    flippedPoints.append(QPointF(src.m().cols-point.x(),src.m().rows-point.y()));
                }
            }
        }

        QList<QRectF> flippedRects;
        foreach(const QRectF &rect, src.file.rects()) {
            if (axis == Y) {
                flippedRects.append(QRectF(src.m().cols-rect.right(),
                                           rect.y(),
                                           rect.width(),
                                           rect.height()));
            } else if (axis == X) {
                flippedRects.append(QRectF(rect.x(),
                                           src.m().rows-rect.bottom(),
                                           rect.width(),
                                           rect.height()));
            } else {
                flippedRects.append(QRectF(src.m().cols-rect.right(),
                                           src.m().rows-rect.bottom(),
                                           rect.width(),
                                           rect.height()));
            }
        }

        dst.file.setPoints(flippedPoints);
        dst.file.setRects(flippedRects);
    }
};

BR_REGISTER(Transform, FlipTransform)

} // namespace br

#include "imgproc/flip.moc"
