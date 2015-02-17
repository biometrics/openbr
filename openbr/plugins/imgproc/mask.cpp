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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Applies an eliptical mask
 * \author Josh Klontz \cite jklontz
 */
class MaskTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        Mat mask(m.size(), CV_8UC1);
        mask.setTo(1);
        const float SCALE = 1.1;
        ellipse(mask, RotatedRect(Point2f(m.cols/2, m.rows/2), Size2f(SCALE*m.cols, SCALE*m.rows), 0), 0, -1);
        dst = m.clone();
        dst.m().setTo(0, mask);
    }
};

BR_REGISTER(Transform, MaskTransform)

} // namespace br

#include "imgproc/mask.moc"
