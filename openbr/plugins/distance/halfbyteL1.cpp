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
#include <openbr/core/distance_sse.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup distances
 * \brief Fast 4-bit L1 distance
 * \author Josh Klontz \cite jklontz
 */
class HalfByteL1Distance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const Mat &a, const Mat &b) const
    {
        return packed_l1(a.data, b.data, a.total());
    }
};

BR_REGISTER(Distance, HalfByteL1Distance)


} // namespace br

#include "distance/halfbyteL1.moc"
