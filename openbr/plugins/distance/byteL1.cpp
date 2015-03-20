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

namespace br
{

/*!
 * \ingroup distances
 * \brief Fast 8-bit L1 distance
 * \author Josh Klontz \cite jklontz
 */
class ByteL1Distance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const unsigned char *a, const unsigned char *b, size_t size) const
    {
        return l1(a, b, size);
    }
};

BR_REGISTER(Distance, ByteL1Distance)

} // namespace br

#include "distance/byteL1.moc"
