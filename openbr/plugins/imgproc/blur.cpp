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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Gaussian blur
 * \author Josh Klontz \cite jklontz
 */
class BlurTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float sigma READ get_sigma WRITE set_sigma RESET reset_sigma STORED false)
    Q_PROPERTY(bool ROI READ get_ROI WRITE set_ROI RESET reset_ROI STORED false)
    BR_PROPERTY(float, sigma, 1)
    BR_PROPERTY(bool, ROI, false)

    void project(const Template &src, Template &dst) const
    {
        if (!ROI) GaussianBlur(src, dst, Size(0,0), sigma);
        else {
            dst.m() = src.m();
            foreach (const QRectF &rect, src.file.rects()) {
                Rect region(rect.x(), rect.y(), rect.width(), rect.height());
                Mat input = dst.m();
                Mat output = input.clone();
                GaussianBlur(input(region), output(region), Size(0,0), sigma);
                dst.m() = output;
            }
        }
    }
};

BR_REGISTER(Transform, BlurTransform)

} // namespace br

#include "imgproc/blur.moc"
