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

#include <QtConcurrent>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Sum match scores across multiple Distances
 * \author Scott Klum \cite sklum
 */
class SumDistance : public ListDistance
{
    Q_OBJECT

    void train(const TemplateList &data)
    {
        QFutureSynchronizer<void> futures;
        foreach (br::Distance *distance, distances)
            futures.addFuture(QtConcurrent::run(distance, &Distance::train, data));
        futures.waitForFinished();
    }

    float compare(const Template &target, const Template &query) const
    {
        float result = 0;

        foreach (br::Distance *distance, distances) {
            result += distance->compare(target, query);

            if (result == -std::numeric_limits<float>::max())
                return result;
        }

        return result;
    }
};

BR_REGISTER(Distance, SumDistance)

} // namespace br

#include "distance/sum.moc"
