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
 * \brief Colorspace conversion.
 * \author Josh Klontz \cite jklontz
 */
class CvtTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(ColorSpace)
    Q_PROPERTY(ColorSpace colorSpace READ get_colorSpace WRITE set_colorSpace RESET reset_colorSpace STORED false)
    Q_PROPERTY(int channel READ get_channel WRITE set_channel RESET reset_channel STORED false)

public:
    enum ColorSpace { Gray = CV_BGR2GRAY,
                      RGBGray = CV_RGB2GRAY,
                      HLS = CV_BGR2HLS,
                      HSV = CV_BGR2HSV,
                      Lab = CV_BGR2Lab,
                      Luv = CV_BGR2Luv,
                      RGB = CV_BGR2RGB,
                      XYZ = CV_BGR2XYZ,
                      YCrCb = CV_BGR2YCrCb,
                      Color = CV_GRAY2BGR };

private:
    BR_PROPERTY(ColorSpace, colorSpace, Gray)
    BR_PROPERTY(int, channel, -1)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() > 1 || colorSpace == Color) cvtColor(src, dst, colorSpace);
        else dst = src;

        if (channel != -1) {
            std::vector<Mat> mv;
            split(dst, mv);
            dst = mv[channel % (int)mv.size()];
        }
    }
};

BR_REGISTER(Transform, CvtTransform)

} // namespace br

#include "imgproc/cvt.moc"
