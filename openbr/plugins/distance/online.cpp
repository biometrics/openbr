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
 * \brief Online Distance metric to attenuate match scores across multiple frames
 * \author Brendan klare \cite bklare
 */
class OnlineDistance : public UntrainableDistance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(float alpha READ get_alpha WRITE set_alpha RESET reset_alpha STORED false)
    BR_PROPERTY(br::Distance*, distance, NULL)
    BR_PROPERTY(float, alpha, 0.1f)

    mutable QHash<QString,float> scoreHash;
    mutable QMutex mutex;

    float compare(const Template &target, const Template &query) const
    {
        float currentScore = distance->compare(target, query);

        QMutexLocker mutexLocker(&mutex);
        return scoreHash[target.file.name] = (1.0- alpha) * scoreHash[target.file.name] + alpha * currentScore;
    }
};

BR_REGISTER(Distance, OnlineDistance)

} // namespace br

#include "distance/online.moc"
