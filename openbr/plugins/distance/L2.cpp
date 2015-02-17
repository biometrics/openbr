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

#include <Eigen/Dense>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief L2 distance computed using eigen.
 * \author Josh Klontz \cite jklontz
 */
class L2Distance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const cv::Mat &a, const cv::Mat &b) const
    {
        const int size = a.rows * a.cols;
        Eigen::Map<Eigen::VectorXf> aMap((float*)a.data, size);
        Eigen::Map<Eigen::VectorXf> bMap((float*)b.data, size);
        return (aMap-bMap).squaredNorm();
    }
};

BR_REGISTER(Distance, L2Distance)

} // namespace br

#include "distance/L2.moc"
