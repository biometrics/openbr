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
 * \brief Cross validate a Distance metric.
 * \author Josh Klontz \cite jklontz
 */
class CrossValidateDistance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        static const QString key("Partition"); // More efficient to preallocate this
        const int partitionA = a.file.get<int>(key, 0);
        const int partitionB = b.file.get<int>(key, 0);
        return (partitionA != partitionB) ? -std::numeric_limits<float>::max() : 0;
    }
};

BR_REGISTER(Distance, CrossValidateDistance)

} // namespace br

#include "distance/crossvalidate.moc"
