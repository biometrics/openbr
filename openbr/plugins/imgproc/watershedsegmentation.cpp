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
 * \brief Applies watershed segmentation.
 * \author Austin Blanton \cite imaus10
 */
class WatershedSegmentationTransform : public UntrainableTransform
{
    Q_OBJECT
    void project(const Template &src, Template &dst) const
    {
        dst = src;

        Mat mod;
//        adaptiveThreshold(src.m(), src.m(), 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 33, 5);
        threshold(src.m(), mod, 0, 255, THRESH_BINARY+THRESH_OTSU);

        // findContours requires an 8-bit 1-channel image
        // and modifies its source image
        if (mod.depth() != CV_8U) OpenCVUtils::cvtUChar(mod, mod);
        if (mod.channels() != 1) OpenCVUtils::cvtGray(mod, mod);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(mod, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

        // draw the contour delineations as 1,2,3... for input to watershed
        Mat markers = Mat::zeros(mod.size(), CV_32S);
        int compCount=0;
        for (int idx=0; idx>=0; idx=hierarchy[idx][0], compCount++) {
            drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);
        }

        Mat orig = src.m();
        // watershed requires a 3-channel 8-bit image
        if (orig.channels() == 1) cvtColor(orig, orig, CV_GRAY2BGR);
        watershed(orig, markers);
        dst.file.set("SegmentsMask", QVariant::fromValue(markers));
        dst.file.set("NumSegments", compCount);
    }
};

BR_REGISTER(Transform, WatershedSegmentationTransform)

} // namespace br

#include "imgproc/watershedsegmentation.moc"
