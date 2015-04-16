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
 * \brief Returns -log(distance(a,b)+1)
 * \author Josh Klontz \cite jklontz
 */
class NegativeLogPlusOneDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    BR_PROPERTY(br::Distance*, distance, NULL)

    bool trainable()
    {
        return distance->trainable();
    }

    void train(const TemplateList &src)
    {
        distance->train(src);
    }

    float compare(const Template &a, const Template &b) const
    {
        return -log(distance->compare(a,b)+1);
    }

    void store(QDataStream &stream) const
    {
        distance->store(stream);
    }

    void load(QDataStream &stream)
    {
        distance->load(stream);
    }
};

BR_REGISTER(Distance, NegativeLogPlusOneDistance)

} // namespace br

#include "distance/neglogplusone.moc"
