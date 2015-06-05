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
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup distances
 * \brief Bayesian quantization Distance
 * \author Josh Klontz \cite jklontz
 */
class BayesianQuantizationDistance : public Distance
{
    Q_OBJECT

    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    BR_PROPERTY(QString, inputVariable, "Label")

    QVector<float> loglikelihoods;

    static void computeLogLikelihood(const Mat &data, const QList<int> &labels, float *loglikelihood)
    {
        const QList<uchar> vals = OpenCVUtils::matrixToVector<uchar>(data);
        if (vals.size() != labels.size())
            qFatal("Logic error.");

        QVector<quint64> genuines(256, 0), impostors(256,0);
        for (int i=0; i<vals.size(); i++)
            for (int j=i+1; j<vals.size(); j++)
                if (labels[i] == labels[j]) genuines[abs(vals[i]-vals[j])]++;
                else                        impostors[abs(vals[i]-vals[j])]++;

        quint64 totalGenuines(0), totalImpostors(0);
        for (int i=0; i<256; i++) {
            totalGenuines += genuines[i];
            totalImpostors += impostors[i];
        }

        for (int i=0; i<256; i++)
            loglikelihood[i] = log((float(genuines[i]+1)/totalGenuines)/(float(impostors[i]+1)/totalImpostors));
    }

    void train(const TemplateList &src)
    {
        if ((src.first().size() > 1) || (src.first().m().type() != CV_8UC1))
            qFatal("Expected sigle matrix templates of type CV_8UC1!");

        const Mat data = OpenCVUtils::toMat(src.data());
        const QList<int> templateLabels = src.indexProperty(inputVariable);
        loglikelihoods = QVector<float>(data.cols*256, 0);

        QFutureSynchronizer<void> futures;
        for (int i=0; i<data.cols; i++)
            futures.addFuture(QtConcurrent::run(&BayesianQuantizationDistance::computeLogLikelihood, data.col(i), templateLabels, &loglikelihoods.data()[i*256]));
        futures.waitForFinished();
    }

    float compare(const cv::Mat &a, const cv::Mat &b) const
    {
        const uchar *aData = a.data;
        const uchar *bData = b.data;
        const int size = a.rows * a.cols;
        float likelihood = 0;
        for (int i=0; i<size; i++)
            likelihood += loglikelihoods[i*256+abs(aData[i]-bData[i])];
        return likelihood;
    }

    void store(QDataStream &stream) const
    {
        stream << loglikelihoods;
    }

    void load(QDataStream &stream)
    {
        stream >> loglikelihoods;
    }
};

BR_REGISTER(Distance, BayesianQuantizationDistance)

} // namespace br

#include "distance/bayesianquantization.moc"
