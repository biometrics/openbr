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
 * \brief Computes magnitude and/or angle of image.
 * \author Josh Klontz \cite jklontz
 */
class GradientTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Channel)
    Q_PROPERTY(Channel channel READ get_channel WRITE set_channel RESET reset_channel STORED false)

public:
    enum Channel { Magnitude, Angle, MagnitudeAndAngle };

private:
    BR_PROPERTY(Channel, channel, Angle)

    void project(const Template &src, Template &dst) const
    {
        Mat dx, dy, magnitude, angle;
        Sobel(src, dx, CV_32F, 1, 0, CV_SCHARR);
        Sobel(src, dy, CV_32F, 0, 1, CV_SCHARR);
        cartToPolar(dx, dy, magnitude, angle, true);
        std::vector<Mat> mv;
        if ((channel == Magnitude) || (channel == MagnitudeAndAngle)) {
            const float theoreticalMaxMagnitude = sqrt(2*pow(float(2*(3+10+3)*255), 2.f));
            mv.push_back(magnitude / theoreticalMaxMagnitude);
        }
        if ((channel == Angle) || (channel == MagnitudeAndAngle))
            mv.push_back(angle);
        Mat result;
        merge(mv, result);
        dst.append(result);
    }
};

BR_REGISTER(Transform, GradientTransform)

} // namespace br

#include "imgproc/gradient.moc"
