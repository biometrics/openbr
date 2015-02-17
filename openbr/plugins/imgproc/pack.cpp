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
 * \brief Compress two uchar into one uchar.
 * \author Josh Klontz \cite jklontz
 */
class PackTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        if ((m.cols % 2 != 0) || (m.type() != CV_8UC1))
            qFatal("Invalid template format.");

        Mat n(m.rows, m.cols/2, CV_8UC1);
        for (int i=0; i<m.rows; i++)
            for (int j=0; j<m.cols/2; j++)
                n.at<uchar>(i,j) = ((m.at<uchar>(i,2*j+0) >> 4) << 4) +
                                   ((m.at<uchar>(i,2*j+1) >> 4) << 0);
        dst = n;
    }
};

BR_REGISTER(Transform, PackTransform)

} // namespace br

#include "imgproc/pack.moc"
