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
 * \brief Split a multi-channel matrix into several single-channel matrices.
 * \author Josh Klontz \cite jklontz
 */
class SplitChannelsTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        std::vector<Mat> mv;
        split(src, mv);
        foreach (const Mat &m, mv)
            dst += m;
    }
};

BR_REGISTER(Transform, SplitChannelsTransform)

/*!
 * \ingroup transforms
 * \brief Split a multi-row matrix into several single-row or smaller multi-row matrices.
 * \br_property int step The numbers of rows to include in each output matrix. 
 * \author Josh Klontz \cite jklontz
 */
class SplitRowsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int step READ get_step WRITE set_step RESET reset_step STORED false)
    Q_PROPERTY(bool cols READ get_cols WRITE set_cols RESET reset_cols STORED false)
    BR_PROPERTY(int, step, 1)
    BR_PROPERTY(bool, cols, false)

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        if (cols) {
            for (int i=0; i<m.cols; i += step)
                dst += m.colRange(i, i + step);
        } else {
            for (int i=0; i<m.rows; i += step)
                dst += m.rowRange(i, i + step);
        }
    }
};

BR_REGISTER(Transform, SplitRowsTransform)

} // namespace br

#include "imgproc/split.moc"
