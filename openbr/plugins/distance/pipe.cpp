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
 * \brief Distances in series.
 * \author Josh Klontz \cite jklontz
 *
 * The templates are compared using each br::Distance in order.
 * If the result of the comparison with any given distance is -FLOAT_MAX then this result is returned early.
 * Otherwise the returned result is the value of comparing the templates using the last br::Distance.
 */
class PipeDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(QList<br::Distance*> distances READ get_distances WRITE set_distances RESET reset_distances)
    BR_PROPERTY(QList<br::Distance*>, distances, QList<br::Distance*>())

    bool trainable()
    {
        for (int i=0; i<distances.size(); i++)
            if (distances[i]->trainable())
                return true;
        return false;
    }

    void train(const TemplateList &data)
    {
        QFutureSynchronizer<void> futures;
        foreach (br::Distance *distance, distances)
            futures.addFuture(QtConcurrent::run(distance, &Distance::train, data));
        futures.waitForFinished();
    }

    float compare(const Template &a, const Template &b) const
    {
        float result = -std::numeric_limits<float>::max();
        foreach (br::Distance *distance, distances) {
            result = distance->compare(a, b);
            if (result == -std::numeric_limits<float>::max())
                return result;
        }
        return result;
    }
};

BR_REGISTER(Distance, PipeDistance)

} // namespace br

#include "distance/pipe.moc"
