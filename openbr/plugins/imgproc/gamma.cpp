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
 * \brief Gamma correction
 * \author Josh Klontz \cite jklontz
 */
class GammaTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float gamma READ get_gamma WRITE set_gamma RESET reset_gamma STORED false)
    BR_PROPERTY(float, gamma, 0.2)

    Mat lut;

    void init()
    {
        lut.create(256, 1, CV_32FC1);
        if (gamma == 0) for (int i=0; i<256; i++) lut.at<float>(i,0) = log((float)i);
        else            for (int i=0; i<256; i++) lut.at<float>(i,0) = pow(i, gamma);
    }

    void project(const Template &src, Template &dst) const
    {
        if (src.m().depth() == CV_8U) LUT(src, lut, dst);
        else                          pow(src, gamma, dst);
    }
};

BR_REGISTER(Transform, GammaTransform)

} // namespace br

#include "imgproc/gamma.moc"
