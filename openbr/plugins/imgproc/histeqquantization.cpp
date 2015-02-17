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
 * \ingroup transforms
 * \brief Approximate floats as uchar with different scalings for each dimension.
 * \author Josh Klontz \cite jklontz
 */
class HistEqQuantizationTransform : public Transform
{
    Q_OBJECT
    QVector<float> thresholds;

    static void computeThresholds(const Mat &data, float *thresholds)
    {
        QList<float> vals = OpenCVUtils::matrixToVector<float>(data);
        std::sort(vals.begin(), vals.end());
        for (int i=0; i<255; i++)
            thresholds[i] = vals[(i+1)*vals.size()/256];
        thresholds[255] = std::numeric_limits<float>::max();
    }

    void train(const TemplateList &src)
    {
        const Mat data = OpenCVUtils::toMat(src.data());
        thresholds = QVector<float>(256*data.cols);

        QFutureSynchronizer<void> futures;
        for (int i=0; i<data.cols; i++)
            futures.addFuture(QtConcurrent::run(&HistEqQuantizationTransform::computeThresholds, data.col(i), &thresholds.data()[i*256]));
        futures.waitForFinished();
    }

    void project(const Template &src, Template &dst) const
    {
        const QList<float> vals = OpenCVUtils::matrixToVector<float>(src);
        dst = Mat(1, vals.size(), CV_8UC1);
        for (int i=0; i<vals.size(); i++) {
            const float *t = &thresholds.data()[i*256];
            const float val = vals[i];
            uchar j = 0;
            while (val > t[j]) j++;
            dst.m().at<uchar>(0,i) = j;
        }
    }

    void store(QDataStream &stream) const
    {
        stream << thresholds;
    }

    void load(QDataStream &stream)
    {
        stream >> thresholds;
    }
};

BR_REGISTER(Transform, HistEqQuantizationTransform)

} // namespace br

#include "imgproc/histeqquantization.moc"
