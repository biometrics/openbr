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
#include <openbr/core/opencvutils.h>
#include <openbr/core/qtutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Selects a random region about a point to use for training.
 * \author Scott Klum \cite sklum
 */

class RndSampleTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int pointIndex READ get_pointIndex WRITE set_pointIndex RESET reset_pointIndex)
    Q_PROPERTY(int sampleRadius READ get_sampleRadius WRITE set_sampleRadius RESET reset_sampleRadius)
    Q_PROPERTY(int sampleFactor READ get_sampleFactor WRITE set_sampleFactor RESET reset_sampleFactor)
    Q_PROPERTY(bool duplicatePositiveSamples READ get_duplicatePositiveSamples WRITE set_duplicatePositiveSamples RESET reset_duplicatePositiveSamples STORED true)
    Q_PROPERTY(QList<float> sampleOverlapBands READ get_sampleOverlapBands WRITE set_sampleOverlapBands RESET reset_sampleOverlapBands STORED true)
    Q_PROPERTY(int samplesPerOverlapBand READ get_samplesPerOverlapBand WRITE set_samplesPerOverlapBand RESET reset_samplesPerOverlapBand STORED true)
    Q_PROPERTY(bool classification READ get_classification WRITE set_classification RESET reset_classification STORED true)
    Q_PROPERTY(float overlapPower READ get_overlapPower WRITE set_overlapPower RESET reset_overlapPower STORED true)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED true)
    BR_PROPERTY(int, pointIndex, 0)
    BR_PROPERTY(int, sampleRadius, 6)
    BR_PROPERTY(int, sampleFactor, 4)
    BR_PROPERTY(bool, duplicatePositiveSamples, true)
    BR_PROPERTY(QList<float>, sampleOverlapBands, QList<float>() << .1 << .5 << .7 << .9 << 1.0)
    BR_PROPERTY(int, samplesPerOverlapBand, 2)
    BR_PROPERTY(bool, classification, true)
    BR_PROPERTY(float, overlapPower, 1)
    BR_PROPERTY(QString, inputVariable, "Label")

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach(const Template &t, src) {
            QPointF point = t.file.points()[pointIndex];
            QRectF region(point.x()-sampleRadius, point.y()-sampleRadius, sampleRadius*2, sampleRadius*2);

            if (region.x() < 0 ||
                region.y() < 0 ||
                region.x() + region.width() >= t.m().cols ||
                region.y() + region.height() >= t.m().rows)
                continue;

            int positiveSamples = duplicatePositiveSamples ? (sampleOverlapBands.size()-1)*samplesPerOverlapBand : 1;
            for (int k=0; k<positiveSamples; k++) {
                dst.append(Template(t.file, Mat(t.m(), OpenCVUtils::toRect(region))));
                dst.last().file.set(inputVariable, 1);
            }

            QList<int> labelCount;
            for (int k=0; k<sampleOverlapBands.size()-1; k++)
                labelCount << 0;

            while (std::accumulate(labelCount.begin(),labelCount.end(),0.0) < (sampleOverlapBands.size()-1)*samplesPerOverlapBand) {

                float x = rand() % (sampleFactor*sampleRadius) + region.x() - sampleFactor/2*sampleRadius;
                float y = rand() % (sampleFactor*sampleRadius) + region.y() - sampleFactor/2*sampleRadius;

                if (x < 0 || y < 0 || x+2*sampleRadius > t.m().cols || y+2*sampleRadius > t.m().rows)
                    continue;

                QRectF negativeLocation = QRectF(x, y, sampleRadius*2, sampleRadius*2);

                float overlap = pow(QtUtils::overlap(region, negativeLocation),overlapPower);

                for (int k = 0; k<sampleOverlapBands.size()-1; k++) {
                    if (overlap >= sampleOverlapBands.at(k) && overlap < sampleOverlapBands.at(k+1) && labelCount[k] < samplesPerOverlapBand) {
                        Mat m(t.m(),OpenCVUtils::toRect(negativeLocation));
                        dst.append(Template(t.file,  m));
                        float label = classification ? 0 : overlap;
                        dst.last().file.set(inputVariable, label);
                        labelCount[k]++;
                    }
                }
            }
        }
    }

    void project(const Template &src, Template &dst) const {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

};

BR_REGISTER(Transform, RndSampleTransform)

} // namespace br

#include "imgproc/rndsample.moc"
