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
#include <openbr/core/common.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Impostor Uniqueness Measure
 * \author Josh Klontz \cite jklontz
 */
class ImpostorUniquenessMeasureTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(double mean READ get_mean WRITE set_mean RESET reset_mean)
    Q_PROPERTY(double stddev READ get_stddev WRITE set_stddev RESET reset_stddev)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    BR_PROPERTY(br::Distance*, distance, Distance::make("Dist(L2)", this))
    BR_PROPERTY(double, mean, 0)
    BR_PROPERTY(double, stddev, 1)
    BR_PROPERTY(QString, inputVariable, "Label")

    TemplateList impostors;

    float calculateIUM(const Template &probe, const TemplateList &gallery) const
    {
        const QString probeLabel = probe.file.get<QString>(inputVariable);
        TemplateList subset = gallery;
        for (int j=subset.size()-1; j>=0; j--)
            if (subset[j].file.get<QString>(inputVariable) == probeLabel)
                subset.removeAt(j);

        QList<float> scores = distance->compare(subset, probe);
        float min, max;
        Common::MinMax(scores, &min, &max);
        double mean = Common::Mean(scores);
        return (max-mean)/(max-min);
    }

    void train(const TemplateList &data)
    {
        distance->train(data);
        impostors = data;

        QList<float> iums; iums.reserve(impostors.size());
        for (int i=0; i<data.size(); i++)
            iums.append(calculateIUM(impostors[i], impostors));

        Common::MeanStdDev(iums, &mean, &stddev);
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        float ium = calculateIUM(src, impostors);
        dst.file.set("Impostor_Uniqueness_Measure", ium);
        dst.file.set("Impostor_Uniqueness_Measure_Bin", ium < mean-stddev ? 0 : (ium < mean+stddev ? 1 : 2));
    }

    void store(QDataStream &stream) const
    {
        distance->store(stream);
        stream << mean << stddev << impostors;
    }

    void load(QDataStream &stream)
    {
        distance->load(stream);
        stream >> mean >> stddev >> impostors;
    }
};

BR_REGISTER(Transform, ImpostorUniquenessMeasureTransform)

} // namespace br

#include "metadata/imposteruniquenessmeasure.moc"
