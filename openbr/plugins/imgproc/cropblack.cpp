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
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Crop out black borders
 * \author Josh Klontz \cite jklontz
 */
class CropBlackTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        Mat gray;
        OpenCVUtils::cvtGray(src, gray);

        int xStart = 0;
        while (xStart < gray.cols) {
            if (mean(gray.col(xStart))[0] >= 1) break;
            xStart++;
        }

        int xEnd = gray.cols - 1;
        while (xEnd >= 0) {
            if (mean(gray.col(xEnd))[0] >= 1) break;
            xEnd--;
        }

        int yStart = 0;
        while (yStart < gray.rows) {
            if (mean(gray.col(yStart))[0] >= 1) break;
            yStart++;
        }

        int yEnd = gray.rows - 1;
        while (yEnd >= 0) {
            if (mean(gray.col(yEnd))[0] >= 1) break;
            yEnd--;
        }

        dst = src.m()(Rect(xStart, yStart, xEnd-xStart, yEnd-yStart));
    }
};

BR_REGISTER(Transform, CropBlackTransform)

} // namespace br

#include "imgproc/cropblack.moc"
