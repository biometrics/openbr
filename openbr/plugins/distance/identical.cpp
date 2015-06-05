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
 * \ingroup distances
 * \brief Returns true if the Templates are identical, false otherwise.
 * \author Josh Klontz \cite jklontz
 */
class IdenticalDistance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const Mat &a, const Mat &b) const
    {
        const size_t size = a.total() * a.elemSize();
        if (size != b.total() * b.elemSize()) return 0;
        for (size_t i=0; i<size; i++)
            if (a.data[i] != b.data[i]) return 0;
        return 1;
    }
};

BR_REGISTER(Distance, IdenticalDistance)

} // namespace br

#include "distance/identical.moc"
