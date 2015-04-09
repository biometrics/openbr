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

#include <numeric>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Fuses similarity scores across multiple matrices of compared templates
 * \author Scott Klum \cite sklum
 * \note Operation: Mean, sum, min, max are supported.
 */
class FuseDistance : public Distance
{
    Q_OBJECT
    Q_ENUMS(Operation)
    Q_PROPERTY(QStringList descriptions READ get_descriptions WRITE set_descriptions RESET reset_descriptions STORED false)
    Q_PROPERTY(Operation operation READ get_operation WRITE set_operation RESET reset_operation STORED false)
    Q_PROPERTY(QList<float> weights READ get_weights WRITE set_weights RESET reset_weights STORED false)

    QList<br::Distance*> distances;

public:
    /*!< */
    enum Operation {Mean, Sum, Max, Min};

private:
    BR_PROPERTY(QStringList, descriptions, QStringList() << "L2")
    BR_PROPERTY(Operation, operation, Mean)
    BR_PROPERTY(QList<float>, weights, QList<float>())

    void init()
    {
        for (int i=0; i<descriptions.size(); i++) {
            distances.append(make(descriptions[i]));
        }
    }

    void train(const TemplateList &src)
    {
        // Partition the templates by matrix
        QList<int> split;
        for (int i=0; i<src.at(0).size(); i++) split.append(1);

        QList<TemplateList> partitionedSrc = src.partition(split);

        // Train on each of the partitions
        for (int i=0; i<descriptions.size(); i++) {
            distances.append(make(descriptions[i]));
            distances[i]->train(partitionedSrc[i]);
        }
    }

    float compare(const Template &a, const Template &b) const
    {
        if (a.size() != distances.size() ||
            b.size() != distances.size())
            return -std::numeric_limits<float>::max();

        QList<float> scores;
        for (int i=0; i<distances.size(); i++) {
            float weight;
            weights.isEmpty() ? weight = 1. : weight = weights[i];
            scores.append(weight*distances[i]->compare(Template(a.file, a[i]),Template(b.file, b[i])));
        }

        switch (operation) {
          case Mean:
            return std::accumulate(scores.begin(),scores.end(),0.0)/(float)scores.size();
            break;
          case Sum:
            return std::accumulate(scores.begin(),scores.end(),0.0);
            break;
          case Min:
            return *std::min_element(scores.begin(),scores.end());
            break;
          case Max:
            return *std::max_element(scores.begin(),scores.end());
            break;
          default:
            qFatal("Invalid operation.");
        }
        return 0;
    }

    void store(QDataStream &stream) const
    {
        stream << distances.size();
        foreach (Distance *distance, distances)
            distance->store(stream);
    }

    void load(QDataStream &stream)
    {
        int numDistances;
        stream >> numDistances;
        for (int i=0; i<descriptions.size(); i++) {
            distances.append(make(descriptions[i]));
            distances[i]->load(stream);
        }

    }
};

BR_REGISTER(Distance, FuseDistance)

} // namespace br

#include "distance/fuse.moc"
