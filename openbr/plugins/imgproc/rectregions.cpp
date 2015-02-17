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
 * \brief Subdivide matrix into rectangular subregions.
 * \author Josh Klontz \cite jklontz
 */
class RectRegionsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int width READ get_width WRITE set_width RESET reset_width STORED false)
    Q_PROPERTY(int height READ get_height WRITE set_height RESET reset_height STORED false)
    Q_PROPERTY(int widthStep READ get_widthStep WRITE set_widthStep RESET reset_widthStep STORED false)
    Q_PROPERTY(int heightStep READ get_heightStep WRITE set_heightStep RESET reset_heightStep STORED false)
    BR_PROPERTY(int, width, 8)
    BR_PROPERTY(int, height, 8)
    BR_PROPERTY(int, widthStep, -1)
    BR_PROPERTY(int, heightStep, -1)

    void project(const Template &src, Template &dst) const
    {
        const int widthStep = this->widthStep == -1 ? width : this->widthStep;
        const int heightStep = this->heightStep == -1 ? height : this->heightStep;
        const Mat &m = src;
        const int xMax = m.cols - width;
        const int yMax = m.rows - height;
        for (int x=0; x <= xMax; x += widthStep)
            for (int y=0; y <= yMax; y += heightStep)
                dst += m(Rect(x, y, width, height));
    }
};

BR_REGISTER(Transform, RectRegionsTransform)

} // namespace br

#include "imgproc/rectregions.moc"
