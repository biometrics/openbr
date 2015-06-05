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
#include <opencv2/imgproc/imgproc_c.h>
#include <limits>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief DOCUMENT ME
 * \br_paper Tan, Xiaoyang, and Bill Triggs.
 *           "Enhanced local texture feature sets for face recognition under difficult lighting conditions."
 *           Analysis and Modeling of Faces and Gestures. Springer Berlin Heidelberg, 2007. 168-182.
 * \author Brendan Klare \cite bklare
 * \author Josh Klontz \cite jklontz
 */
class LTPTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int radius READ get_radius WRITE set_radius RESET reset_radius STORED false)
    Q_PROPERTY(float threshold READ get_threshold WRITE set_threshold RESET reset_threshold STORED false)
    BR_PROPERTY(int,   radius,    1)
    BR_PROPERTY(float, threshold, 0.1F)

    unsigned short lut[8][3];
    uchar null;

    void init()
    {
        unsigned short cnt = 0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 3; j++) 
                lut[i][j] = cnt++;
            cnt++;  //we skip the 4th number (only three patterns)
        }
    }

    void project(const Template &src, Template &dst) const
    {
        Mat m; src.m().convertTo(m, CV_32F); assert(m.isContinuous() && (m.channels() == 1));

        Mat n(m.rows, m.cols, CV_16U);
        n = null; 
        float thresholdNeg = -1.0 * threshold; //compute once (can move to init)

        const float *p = (const float*)m.ptr();
        float diff;
        for (int r=radius; r<m.rows-radius; r++) {
            for (int c=radius; c<m.cols-radius; c++) {
                const float cval  = (p[(r+0*radius)*m.cols+c+0*radius]);

                diff = p[(r-1*radius)*m.cols+c-1*radius] - cval;
                if      (diff > threshold)      n.at<unsigned short>(r,c) = lut[0][0];
                else if (diff < thresholdNeg)   n.at<unsigned short>(r,c) = lut[0][1];
                else                            n.at<unsigned short>(r,c) = lut[0][2];

                diff = p[(r-1*radius)*m.cols+c+0*radius] - cval;
                if      (diff > threshold)      n.at<unsigned short>(r,c) += lut[1][0];
                else if (diff < thresholdNeg)   n.at<unsigned short>(r,c) += lut[1][1];
                else                            n.at<unsigned short>(r,c) += lut[1][2];

                diff = p[(r-1*radius)*m.cols+c+1*radius] - cval;
                if      (diff > threshold)      n.at<unsigned short>(r,c) += lut[2][0];
                else if (diff < thresholdNeg)   n.at<unsigned short>(r,c) += lut[2][1];
                else                            n.at<unsigned short>(r,c) += lut[2][2];

                diff = p[(r+0*radius)*m.cols+c+1*radius] - cval;
                if      (diff > threshold)      n.at<unsigned short>(r,c) += lut[3][0];
                else if (diff < thresholdNeg)   n.at<unsigned short>(r,c) += lut[3][1];
                else                            n.at<unsigned short>(r,c) += lut[3][2];

                diff = p[(r+1*radius)*m.cols+c+1*radius] - cval;
                if      (diff > threshold)      n.at<unsigned short>(r,c) += lut[4][0];
                else if (diff < thresholdNeg)   n.at<unsigned short>(r,c) += lut[4][1];
                else                            n.at<unsigned short>(r,c) += lut[4][2];

                diff = p[(r+1*radius)*m.cols+c+0*radius] - cval;
                if      (diff > threshold)      n.at<unsigned short>(r,c) += lut[5][0];
                else if (diff < thresholdNeg)   n.at<unsigned short>(r,c) += lut[5][1];
                else                            n.at<unsigned short>(r,c) += lut[5][2];

                diff = p[(r+1*radius)*m.cols+c-1*radius] - cval;
                if      (diff > threshold)      n.at<unsigned short>(r,c) += lut[6][0];
                else if (diff < thresholdNeg)   n.at<unsigned short>(r,c) += lut[6][1];
                else                            n.at<unsigned short>(r,c) += lut[6][2];

                diff = p[(r+0*radius)*m.cols+c-1*radius] - cval;
                if      (diff > threshold)      n.at<unsigned short>(r,c) += lut[7][0];
                else if (diff < thresholdNeg)   n.at<unsigned short>(r,c) += lut[7][1];
                else                            n.at<unsigned short>(r,c) += lut[7][2];
            }
        }

        dst += n;
    }
};

BR_REGISTER(Transform, LTPTransform)

} // namespace br

#include "imgproc/ltp.moc"
