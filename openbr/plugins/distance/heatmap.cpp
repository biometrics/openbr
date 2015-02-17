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
 * \brief 1v1 heat map comparison
 * \author Scott Klum \cite sklum
 */
class HeatMapDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description STORED false)
    Q_PROPERTY(int step READ get_step WRITE set_step RESET reset_step STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    BR_PROPERTY(QString, description, "IdenticalDistance")
    BR_PROPERTY(int, step, 1)
    BR_PROPERTY(QString, inputVariable, "Label")

    QList<br::Distance*> distances;

    void train(const TemplateList &src)
    {
        QList<TemplateList> patches;

        // Split src into list of TemplateLists of corresponding patches across all Templates
        for (int i=0; i<step; i++) {
            TemplateList patchBuffer;
            for (int j=0; j<src.size(); j++)
                patchBuffer.append(Template(src[j].file, src[j][i]));
            patches.append(patchBuffer);
            patchBuffer.clear();
        }

        while (distances.size() < patches.size())
            distances.append(make(description));

        // Train on a distance for each patch
        for (int i=0; i<distances.size(); i++)
            distances[i]->train(patches[i]);
    }

    float compare(const cv::Mat &target, const cv::Mat &query) const
    {
        (void) target;
        (void) query;
        qFatal("Heatmap Distance not compatible with Template to Template comparison.");

        return 0;
    }

    void compare(const TemplateList &target, const TemplateList &query, Output *output) const
    {
        for (int i=0; i<target.size(); i++) {
            if (target[i].size() != step || query[i].size() != step) qFatal("Heatmap step not equal to the number of patches.");
            for (int j=0; j<step; j++)
                output->setRelative(distances[j]->compare(target[i][j],query[i][j]), j, 0);
        }
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
        while (distances.size() < numDistances)
            distances.append(make(description));
        foreach (Distance *distance, distances)
            distance->load(stream);
    }
};

BR_REGISTER(Distance, HeatMapDistance)

} // namespace br

#include "distance/heatmap.moc"
