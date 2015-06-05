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

#include <opencv2/photo/photo.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV Non-Local Means Denoising
 * \br_link http://docs.opencv.org/modules/photo/doc/denoising.html
 * \author Josh Klontz \cite jklontz
 */
class NLMeansDenoisingTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float h READ get_h WRITE set_h RESET reset_h STORED false)
    Q_PROPERTY(int templateWindowSize READ get_templateWindowSize WRITE set_templateWindowSize RESET reset_templateWindowSize STORED false)
    Q_PROPERTY(int searchWindowSize READ get_searchWindowSize WRITE set_searchWindowSize RESET reset_searchWindowSize STORED false)
    BR_PROPERTY(float, h, 3)
    BR_PROPERTY(int, templateWindowSize, 7)
    BR_PROPERTY(int, searchWindowSize, 21)

    void project(const Template &src, Template &dst) const
    {
        fastNlMeansDenoising(src, dst, h, templateWindowSize, searchWindowSize);
    }
};

BR_REGISTER(Transform, NLMeansDenoisingTransform)

} // namespace br

#include "imgproc/denoising.moc"
