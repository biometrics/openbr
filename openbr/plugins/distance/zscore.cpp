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
 * \brief Performs zscore normalization on distances at test time by learning mean
 *        and standard deviation parameters during training.
 * \author Scott Klum \cite sklum
 */
class ZScoreDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(bool crossModality READ get_crossModality WRITE set_crossModality RESET reset_crossModality STORED false)
    BR_PROPERTY(br::Distance*, distance, make("Dist(L2)"))
    BR_PROPERTY(bool, crossModality, false)

    float min, max;
    double mean, stddev;

    void train(const TemplateList &src)
    {
        distance->train(src);

        QScopedPointer<MatrixOutput> matrixOutput(MatrixOutput::make(FileList(src.size()), FileList(src.size())));
        distance->compare(src, src, matrixOutput.data());

        QList<float> scores;
        scores.reserve(src.size()*src.size());
        for (int i=0; i<src.size(); i++) {
            for (int j=0; j<i; j++) {
                const float score = matrixOutput.data()->data.at<float>(i, j);
                if (score == -std::numeric_limits<float>::max()) continue;
                if (crossModality && src[i].file.get<QString>("MODALITY") == src[j].file.get<QString>("MODALITY")) continue;
                scores.append(score);
            }
        }

        Common::MinMax(scores, &min, &max);
        Common::MeanStdDev(scores, &mean, &stddev);

        if (stddev == 0) qFatal("Stddev is 0.");
    }

    float compare(const Template &target, const Template &query) const
    {
        float score = distance->compare(target,query);
        if      (score == -std::numeric_limits<float>::max()) score = (min - mean) / stddev;
        else if (score ==  std::numeric_limits<float>::max()) score = (max - mean) / stddev;
        else                                                  score = (score - mean) / stddev;
        return score;
    }

    void store(QDataStream &stream) const
    {
        distance->store(stream);
        stream << min << max << mean << stddev;
    }

    void load(QDataStream &stream)
    {
        distance->load(stream);
        stream >> min >> max >> mean >> stddev;
    }
};

BR_REGISTER(Distance, ZScoreDistance)

} // namespace br

#include "distance/zscore.moc"
