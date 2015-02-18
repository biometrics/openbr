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
 * \brief Divide the matrix into 4 smaller matricies of equal size.
 * \author Josh Klontz \cite jklontz
 */
class SubdivideTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        const int subrows = m.rows/2;
        const int subcolumns = m.cols/2;
        dst.append(Mat(m,Rect(0,          0, subcolumns, subrows)).clone());
        dst.append(Mat(m,Rect(subcolumns, 0, subcolumns, subrows)).clone());
        dst.append(Mat(m,Rect(0,          subrows, subcolumns, subrows)).clone());
        dst.append(Mat(m,Rect(subcolumns, subrows, subcolumns, subrows)).clone());
    }
};

BR_REGISTER(Transform, SubdivideTransform)

} // namespace br

#include "imgproc/subdivide.moc"
