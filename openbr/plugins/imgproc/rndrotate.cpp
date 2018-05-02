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
 * \brief Randomly rotates an image in a specified range.
 * \author Scott Klum \cite sklum
 */
class RndRotateTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QList<int> range READ get_range WRITE set_range RESET reset_range STORED false)
    Q_PROPERTY(bool rotateMat READ get_rotateMat WRITE set_rotateMat RESET reset_rotateMat STORED false)
    Q_PROPERTY(bool rotatePoints READ get_rotatePoints WRITE set_rotatePoints RESET reset_rotatePoints STORED false)
    Q_PROPERTY(bool rotateRects READ get_rotateRects WRITE set_rotateRects RESET reset_rotateRects STORED false)
    Q_PROPERTY(int centerIndex READ get_centerIndex WRITE set_centerIndex RESET reset_centerIndex STORED false)
    Q_PROPERTY(bool useRect READ get_useRect WRITE set_useRect RESET reset_useRect STORED false)   
    BR_PROPERTY(QList<int>, range, QList<int>() << -15 << 15)
    BR_PROPERTY(bool, rotateMat, true)
    BR_PROPERTY(bool, rotatePoints, true)
    BR_PROPERTY(bool, rotateRects, true)
    BR_PROPERTY(int, centerIndex, -1)
    BR_PROPERTY(bool, useRect, false);

    void project(const Template &src, Template &dst) const {
        const int span = range.first() - range.last();
        const int angle = span == 0 ? range.first() : (rand() % span) + range.first();
        const QPointF center = centerIndex == -1 ? QPointF(src.m().rows/2,src.m().cols/2) : (useRect ? src.file.rects()[centerIndex].center() : src.file.points()[centerIndex]);
        OpenCVUtils::rotate(src, dst, angle, rotateMat, rotatePoints, rotateRects, center);
    }
};

BR_REGISTER(Transform, RndRotateTransform)

} // namespace br

#include "imgproc/rndrotate.moc"
