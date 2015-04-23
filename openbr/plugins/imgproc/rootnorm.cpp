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
 * \brief dst=sqrt(norm_L1(src)) proposed as RootSIFT (see paper)
 * \br_paper Arandjelovic, Relja, and Andrew Zisserman.
 *           "Three things everyone should know to improve object retrieval."
 *           Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.
 * \author Josh Klontz \cite jklontz
 */
class RootNormTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        dst.m() = Mat(m.rows, m.cols, m.type());
        for (int i=0; i<m.rows; i++) {
            Mat temp;
            cv::normalize(m.row(i), temp, 1, 0, NORM_L1);
            cv::sqrt(temp, temp);
            temp.copyTo(dst.m().row(i));
        }
    }
};

BR_REGISTER(Transform, RootNormTransform)

} // namespace br

#include "imgproc/rootnorm.moc"
