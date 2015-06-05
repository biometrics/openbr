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
 * \brief Make a mask over skin in an image
 * \br_link http://worldofcameras.wordpress.com/tag/skin-detection-opencv/
 * \author Josh Klontz \cite jklontz
 */
class SkinMaskTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        Mat m;
        cvtColor(src, m, CV_BGR2YCrCb);
        std::vector<Mat> mv;
        split(m, mv);
        Mat mask = Mat(m.rows, m.cols, CV_8UC1);

        for (int i=0; i<m.rows; i++) {
            for (int j=0; j<m.cols; j++) {
                int Cr= mv[1].at<quint8>(i,j);
                int Cb =mv[2].at<quint8>(i,j);
                mask.at<quint8>(i, j) = (Cr>130 && Cr<170) && (Cb>70 && Cb<125) ? 255 : 0;
            }
        }

        dst = mask;
    }
};

BR_REGISTER(Transform, SkinMaskTransform)

} // namespace br

#include "imgproc/skinmask.moc"
