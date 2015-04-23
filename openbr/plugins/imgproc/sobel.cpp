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

#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Approximates the gradient in an image using sobel operator.
 * \author Scott Klum \cite sklum
 */
class SobelTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(int ksize READ get_ksize WRITE set_ksize RESET reset_ksize STORED false)
    Q_PROPERTY(float scale READ get_scale WRITE set_scale RESET reset_scale STORED false)
    BR_PROPERTY(int, ksize, 3)
    BR_PROPERTY(float, scale, 1)

    void project(const Template &src, Template &dst) const
    {
          Mat dx, abs_dx, dy, abs_dy;
          Sobel(src, dx, CV_32F, 1, 0, ksize, scale);
          Sobel(src, dy, CV_32F, 0, 1, ksize, scale);
          convertScaleAbs(dx, abs_dx);
          convertScaleAbs(dy, abs_dy);
          addWeighted(abs_dx, 0.5, abs_dy, 0.5, 0, dst);
    }
};

BR_REGISTER(Transform, SobelTransform)

} // namespace br

#include "imgproc/sobel.moc"


