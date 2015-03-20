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
 * \brief Limit the size of the template
 * \author Josh Klontz \cite jklontz
 */
class LimitSizeTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int max READ get_max WRITE set_max RESET reset_max STORED false)
    BR_PROPERTY(int, max, -1)

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        if (m.rows > m.cols)
            if (m.rows > max) resize(m, dst, Size(std::max(1, m.cols * max / m.rows), max));
            else              dst = m;
        else
            if (m.cols > max) resize(m, dst, Size(max, std::max(1, m.rows * max / m.cols)));
            else              dst = m;
    }
};

BR_REGISTER(Transform, LimitSizeTransform)

} // namespace br

#include "imgproc/limitsize.moc"
