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
 * \brief An integral histogram
 * \author Josh Klontz \cite jklontz
 */
class IntegralHistTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int bins READ get_bins WRITE set_bins RESET reset_bins STORED false)
    Q_PROPERTY(int radius READ get_radius WRITE set_radius RESET reset_radius STORED false)
    BR_PROPERTY(int, bins, 256)
    BR_PROPERTY(int, radius, 16)

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src.m();
        if (m.type() != CV_8UC1) qFatal("IntegralHist requires 8UC1 matrices.");

        Mat integral(m.rows/radius+1, (m.cols/radius+1)*bins, CV_32SC1);
        integral.setTo(0);
        for (int i=1; i<integral.rows; i++) {
            for (int j=1; j<integral.cols; j+=bins) {
                for (int k=0; k<bins; k++) integral.at<qint32>(i, j+k) += integral.at<qint32>(i-1, j     +k);
                for (int k=0; k<bins; k++) integral.at<qint32>(i, j+k) += integral.at<qint32>(i  , j-bins+k);
                for (int k=0; k<bins; k++) integral.at<qint32>(i, j+k) -= integral.at<qint32>(i-1, j-bins+k);
                for (int k=0; k<radius; k++)
                    for (int l=0; l<radius; l++)
                        integral.at<qint32>(i, j+m.at<quint8>((i-1)*radius+k,(j/bins-1)*radius+l))++;
            }
        }
        dst = integral;
    }
};

BR_REGISTER(Transform, IntegralHistTransform)

} // namespace br

#include "imgproc/integralhist.moc"
