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
 * \brief Linear normalizes of a Distance so the mean impostor score is 0 and the mean genuine score is 1.
 * \author Josh Klontz \cite jklontz
 */
class UnitDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance *distance READ get_distance WRITE set_distance RESET reset_distance)
    Q_PROPERTY(float a READ get_a WRITE set_a RESET reset_a)
    Q_PROPERTY(float b READ get_b WRITE set_b RESET reset_b)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    BR_PROPERTY(br::Distance*, distance, make("Dist(L2)"))
    BR_PROPERTY(float, a, 1)
    BR_PROPERTY(float, b, 0)
    BR_PROPERTY(QString, inputVariable, "Label")

    void train(const TemplateList &templates)
    {
        const TemplateList samples = templates.mid(0, 2000);
        const QList<int> sampleLabels = samples.indexProperty(inputVariable);
        QScopedPointer<MatrixOutput> matrixOutput(MatrixOutput::make(FileList(samples.size()), FileList(samples.size())));
        Distance::compare(samples, samples, matrixOutput.data());

        double genuineAccumulator, impostorAccumulator;
        int genuineCount, impostorCount;
        genuineAccumulator = impostorAccumulator = genuineCount = impostorCount = 0;

        for (int i=0; i<samples.size(); i++) {
            for (int j=0; j<i; j++) {
                const float val = matrixOutput.data()->data.at<float>(i, j);
                if (sampleLabels[i] == sampleLabels[j]) {
                    genuineAccumulator += val;
                    genuineCount++;
                } else {
                    impostorAccumulator += val;
                    impostorCount++;
                }
            }
        }

        if (genuineCount == 0) { qWarning("No genuine matches."); return; }
        if (impostorCount == 0) { qWarning("No impostor matches."); return; }

        double genuineMean = genuineAccumulator / genuineCount;
        double impostorMean = impostorAccumulator / impostorCount;

        if (genuineMean == impostorMean) { qWarning("Genuines and impostors are indistinguishable."); return; }

        a = 1.0/(genuineMean-impostorMean);
        b = impostorMean;

        qDebug("a = %f, b = %f", a, b);
    }

    float compare(const Template &target, const Template &query) const
    {
        return normalize(distance->compare(target, query));
    }

    float compare(const cv::Mat &target, const cv::Mat &query) const
    {
        return normalize(distance->compare(target, query));
    }
	
	float normalize(float score) const
    {
        if (!Globals->scoreNormalization) return score;
        return a * (score - b);
    }
};

BR_REGISTER(Distance, UnitDistance)

} // namespace br

#include "distance/unit.moc"
