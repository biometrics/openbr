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
#include <algorithm>

#include <openbr/plugins/openbr_internal.h>

#include <QtConcurrent>

using namespace std;

namespace br
{

/*!
 * \ingroup distances
 * \brief Compares all permutations of matrices from one template to the other, and fuses the scores via the operation specified.
 * \author Scott Klum \cite sklum
 * \note Operation: Mean, sum, min, max are supported.
 */
class PermuteDistance : public Distance
{
    Q_OBJECT
    Q_ENUMS(Operation)
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(Operation operation READ get_operation WRITE set_operation RESET reset_operation STORED false)
public:
    /*!< */
    enum Operation {Mean, Sum, Max, Min};

private:
    BR_PROPERTY(br::Distance*, distance, NULL)
    BR_PROPERTY(Operation, operation, Mean)

    void train(const TemplateList &src)
    {
        distance->train(src);
    }

    float compare(const Template &a, const Template &b) const
    {
        QList<int> indices;
        for (int i=0; i<a.size(); i++)
            indices << i;

        QList<float> scores;
        do {
            QList<float> permScores;
            for (int i=0; i<a.size(); i++)
                permScores.append(distance->compare(Template(a.file, a[indices[i]]),Template(b.file, b[i])));
            scores.append(std::accumulate(permScores.begin(),permScores.end(),0.0));
        } while ( next_permutation(indices.begin(),indices.end()) );

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
};

BR_REGISTER(Distance, PermuteDistance)

} // namespace br

#include "distance/permute.moc"

