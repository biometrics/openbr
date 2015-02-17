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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Masks image according to pixel change.
 * \author Josh Klontz \cite jklontz
 */
class GradientMaskTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int delta READ get_delta WRITE set_delta RESET reset_delta STORED false)
    BR_PROPERTY(int, delta, 1)

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src.m();
        if (m.type() != CV_8UC1) qFatal("Requires 8UC1 matrices.");
        Mat n = Mat(m.rows, m.cols, CV_8UC1);
        n.setTo(255);
        for (int i=0; i<m.rows; i++) {
            for (int j=0; j<m.cols; j++) {
                if ((i>0)        && (abs(m.at<quint8>(i-1,j)-m.at<quint8>(i,j)) > delta)) { n.at<quint8>(i,j) = 0; continue; }
                if ((j+1<m.cols) && (abs(m.at<quint8>(i,j+1)-m.at<quint8>(i,j)) > delta)) { n.at<quint8>(i,j) = 0; continue; }
                if ((i+1<m.rows) && (abs(m.at<quint8>(i+1,j)-m.at<quint8>(i,j)) > delta)) { n.at<quint8>(i,j) = 0; continue; }
                if ((j>0)        && (abs(m.at<quint8>(i,j-1)-m.at<quint8>(i,j)) > delta)) { n.at<quint8>(i,j) = 0; continue; }
            }
        }
        dst = n;
    }
};

BR_REGISTER(Transform, GradientMaskTransform)

} // namespace br

#include "imgproc/gradientmask.moc"
