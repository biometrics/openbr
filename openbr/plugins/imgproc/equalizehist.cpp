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
 * \brief Histogram equalization
 * \author Josh Klontz \cite jklontz
 */
class EqualizeHistTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() == 1) {
            equalizeHist(src, dst);
        } else if (src.m().channels() == 3) {
            // http://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
            Mat ycrcb;
            cvtColor(src, ycrcb, COLOR_BGR2YCrCb);
            std::vector<Mat> channels;
            split(ycrcb, channels);
            equalizeHist(channels[0], channels[0]);
            merge(channels, ycrcb);
            cvtColor(ycrcb, dst, COLOR_YCrCb2BGR);
        } else {
            qFatal("Invalid channel count!");
        }
    }
};

BR_REGISTER(Transform, EqualizeHistTransform)

} // namespace br

#include "imgproc/equalizehist.moc"
