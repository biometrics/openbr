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

#include <opencv2/objdetect/objdetect.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief OpenCV HOGDescriptor wrapper
 * \br_link http://docs.opencv.org/modules/gpu/doc/object_detection.html
 * \author Austin Blanton \cite imaus10
 */
class HoGDescriptorTransform : public UntrainableTransform
{
    Q_OBJECT

    HOGDescriptor hog;

    void project(const Template &src, Template &dst) const
    {
        std::vector<float> descriptorVals;
        std::vector<Point> locations;
        Size winStride = Size(0,0);
        Size padding = Size(0,0);
        foreach (const Mat &rect, src) {
            hog.compute(rect, descriptorVals, winStride, padding, locations);
            Mat HoGFeats(descriptorVals, true);
            dst += HoGFeats;
        }
    }
};

BR_REGISTER(Transform, HoGDescriptorTransform)

} // namespace br

#include "imgproc/hog.moc"
