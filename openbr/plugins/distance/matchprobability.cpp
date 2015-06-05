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
#include <openbr/core/common.h>

namespace br
{

float KDEPointer(const QList<float> *scores, double x, double h)
{
    return Common::KernelDensityEstimation(*scores, x, h);
}

/* Kernel Density Estimator */
struct KDE
{
    float min, max;
    double mean, stddev;
    QList<float> bins;

    KDE() : min(0), max(1), mean(0), stddev(1) {}

    KDE(const QList<float> &scores, bool trainKDE)
    {
        Common::MinMax(scores, &min, &max);
        Common::MeanStdDev(scores, &mean, &stddev);

        if (!trainKDE)
            return;

        double h = Common::KernelDensityBandwidth(scores);
        const int size = 255;
        bins.reserve(size);

        QFutureSynchronizer<float> futures;

        for (int i=0; i < size; i++)
            futures.addFuture(QtConcurrent::run(KDEPointer, &scores, min + (max-min)*i/(size-1), h));
        futures.waitForFinished();

        foreach(const QFuture<float> & future, futures.futures())
            bins.append(future.result());
    }

    float operator()(float score, bool gaussian = true) const
    {
        if (gaussian) return 1/(stddev*sqrt(2*CV_PI))*exp(-0.5*pow((score-mean)/stddev, 2));
        if (bins.empty())
            return -std::numeric_limits<float>::max();

        if (score <= min) return bins.first();
        if (score >= max) return bins.last();
        const float x = (score-min)/(max-min)*bins.size();
        const float y1 = bins[floor(x)];
        const float y2 = bins[ceil(x)];
        return y1 + (y2-y1)*(x-floor(x));
    }
};

QDataStream &operator<<(QDataStream &stream, const KDE &kde)
{
    return stream << kde.min << kde.max << kde.mean << kde.stddev << kde.bins;
}

QDataStream &operator>>(QDataStream &stream, KDE &kde)
{
    return stream >> kde.min >> kde.max >> kde.mean >> kde.stddev >> kde.bins;
}

/* Match Probability */
struct MP
{
    KDE genuine, impostor;
    MP() {}
    MP(const QList<float> &genuineScores, const QList<float> &impostorScores, bool trainKDE)
        : genuine(genuineScores, trainKDE), impostor(impostorScores, trainKDE) {}
    float operator()(float score, bool gaussian = true) const
    {
        const float g = genuine(score, gaussian);
        const float s = g / (impostor(score, gaussian) + g);
        return s;
    }
};

QDataStream &operator<<(QDataStream &stream, const MP &nmp)
{
    return stream << nmp.genuine << nmp.impostor;
}

QDataStream &operator>>(QDataStream &stream, MP &nmp)
{
    return stream >> nmp.genuine >> nmp.impostor;
}

/*!
 * \ingroup distances
 * \brief Match Probability
 * \author Josh Klontz \cite jklontz
 */
class MatchProbabilityDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(bool gaussian READ get_gaussian WRITE set_gaussian RESET reset_gaussian STORED false)
    Q_PROPERTY(bool crossModality READ get_crossModality WRITE set_crossModality RESET reset_crossModality STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)

    MP mp;

    void train(const TemplateList &src)
    {
        distance->train(src);

        const QList<int> labels = src.indexProperty(inputVariable);
        QScopedPointer<MatrixOutput> matrixOutput(MatrixOutput::make(FileList(src.size()), FileList(src.size())));
        distance->compare(src, src, matrixOutput.data());

        QList<float> genuineScores, impostorScores;
        genuineScores.reserve(labels.size());
        impostorScores.reserve(labels.size()*labels.size());
        for (int i=0; i<src.size(); i++) {
            for (int j=0; j<i; j++) {
                const float score = matrixOutput.data()->data.at<float>(i, j);
                if (score == -std::numeric_limits<float>::max()) continue;
                if (crossModality && src[i].file.get<QString>("MODALITY") == src[j].file.get<QString>("MODALITY")) continue;
                if (labels[i] == labels[j]) genuineScores.append(score);
                else                        impostorScores.append(score);
            }
        }

        mp = MP(genuineScores, impostorScores, !gaussian);
    }

    float compare(const Template &target, const Template &query) const
    {
        return normalize(distance->compare(target, query));
    }

    float compare(const cv::Mat &target, const cv::Mat &query) const
    {
        return normalize(distance->compare(target, query));
    }

    float compare(const uchar *a, const uchar *b, size_t size) const
    {
        return normalize(distance->compare(a, b, size));
    }

    float normalize(float score) const
    {
        if (score == -std::numeric_limits<float>::max()) return score;
        if (!Globals->scoreNormalization) return -log(score+1);
        return mp(score, gaussian);
    }

    void store(QDataStream &stream) const
    {
        distance->store(stream);
        stream << mp;
    }

    void load(QDataStream &stream)
    {
        distance->load(stream);
        stream >> mp;
    }

protected:
    BR_PROPERTY(br::Distance*, distance, make("Dist(L2)"))
    BR_PROPERTY(bool, gaussian, true)
    BR_PROPERTY(bool, crossModality, false)
    BR_PROPERTY(QString, inputVariable, "Label")
};

BR_REGISTER(Distance, MatchProbabilityDistance)

} // namespace br

#include "distance/matchprobability.moc"
