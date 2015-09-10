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

#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Creates a bounding square around three points (typically the two eyes and the chin).
 * \author Scott Klum \cite sklum
 * \br_property int eyeL index of left eye point.
 * \br_property int eyeR Index of right eye point.
 * \br_property int chin Index of chin point.
 */
class SquareFromPointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    Q_PROPERTY(int eyeR READ get_eyeR WRITE set_eyeR RESET reset_eyeR STORED false)
    Q_PROPERTY(int eyeL READ get_eyeL WRITE set_eyeL RESET reset_eyeL STORED false)
    Q_PROPERTY(int chin READ get_chin WRITE set_chin RESET reset_chin STORED false)
    BR_PROPERTY(int, eyeL, 7)
    BR_PROPERTY(int, eyeR, 10)
    BR_PROPERTY(int, chin, 20)

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        QList<QPointF> points = src.points();

        float side = points[chin].y() - points[eyeL].y();
        float left = points[eyeL].x()+(points[eyeR].x() - points[eyeL].x())/2.-side/2.;
        QRectF rect(left,points[eyeL].y(),side,side);

        dst.setRects(OpenCVUtils::toRects(QList<QRectF>() << rect));
    }
};

BR_REGISTER(Transform, SquareFromPointsTransform)

} // namespace br

#include "metadata/squarefrompoints.moc"
