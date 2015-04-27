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
 * \brief Trims a percentage of width and height from the border of a matrix.
 * \author Scott Klum \cite sklum
 */
class TrimTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float width READ get_width WRITE set_width RESET reset_width STORED false)
    Q_PROPERTY(float height READ get_height WRITE set_height RESET reset_height STORED false)
    BR_PROPERTY(float, width, .2)
    BR_PROPERTY(float, height, .2)

    void project(const Template &src, Template &dst) const
    {
        dst = Mat(src, Rect(src.m().cols*width/2, src.m().rows*height/2, src.m().cols*(1-width), src.m().rows*(1-height)));
    }
};

BR_REGISTER(Transform, TrimTransform)

} // namespace br

#include "imgproc/trim.moc"
