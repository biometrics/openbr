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

namespace br
{

/*!
 * \ingroup distances
 * \brief DistDistance wrapper.
 * \author Josh Klontz \cite jklontz
 */
class DefaultDistance : public UntrainableDistance
{
    Q_OBJECT
    Distance *distance;

    void init()
    {
        distance = Distance::make("Dist("+file.suffix()+")");
    }

    float compare(const cv::Mat &a, const cv::Mat &b) const
    {
        return distance->compare(a, b);
    }
};

BR_REGISTER(Distance, DefaultDistance)

} // namespace br

#include "distance/default.moc"
