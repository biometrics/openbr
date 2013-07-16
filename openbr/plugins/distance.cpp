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

#include <QFutureSynchronizer>
#include <QtConcurrentRun>
#include <opencv2/imgproc/imgproc.hpp>
#include "openbr_internal.h"

#include "openbr/core/distance_sse.h"
#include "openbr/core/qtutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup distances
 * \brief Standard distance metrics
 * \author Josh Klontz \cite jklontz
 */
class DistDistance : public Distance
{
    Q_OBJECT
    Q_ENUMS(Metric)
    Q_PROPERTY(Metric metric READ get_metric WRITE set_metric RESET reset_metric STORED false)
    Q_PROPERTY(bool negLogPlusOne READ get_negLogPlusOne WRITE set_negLogPlusOne RESET reset_negLogPlusOne STORED false)

public:
    /*!< */
    enum Metric { Correlation,
                  ChiSquared,
                  Intersection,
                  Bhattacharyya,
                  INF,
                  L1,
                  L2,
                  Cosine };

private:
    BR_PROPERTY(Metric, metric, L2)
    BR_PROPERTY(bool, negLogPlusOne, true)

    float compare(const Template &a, const Template &b) const
    {
        if ((a.m().size != b.m().size) ||
            (a.m().type() != b.m().type()))
                return -std::numeric_limits<float>::max();

        float result = std::numeric_limits<float>::max();
        switch (metric) {
          case Correlation:
            return compareHist(a, b, CV_COMP_CORREL);
          case ChiSquared:
            result = compareHist(a, b, CV_COMP_CHISQR);
            break;
          case Intersection:
            result = compareHist(a, b, CV_COMP_INTERSECT);
            break;
          case Bhattacharyya:
            result = compareHist(a, b, CV_COMP_BHATTACHARYYA);
            break;
          case INF:
            result = norm(a, b, NORM_INF);
            break;
          case L1:
            result = norm(a, b, NORM_L1);
            break;
          case L2:
            result = norm(a, b, NORM_L2);
            break;
          case Cosine:
            return cosine(a, b);
          default:
            qFatal("Invalid metric");
        }

        if (result != result)
            qFatal("NaN result.");

        return negLogPlusOne ? -log(result+1) : result;
    }

    static float cosine(const Mat &a, const Mat &b)
    {
        float dot = 0;
        float magA = 0;
        float magB = 0;

        for (int row=0; row<a.rows; row++) {
            for (int col=0; col<a.cols; col++) {
                const float target = a.at<float>(row,col);
                const float query = b.at<float>(row,col);
                dot += target * query;
                magA += target * target;
                magB += query * query;
            }
        }

        return dot / (sqrt(magA)*sqrt(magB));
    }
};

BR_REGISTER(Distance, DistDistance)

/*!
 * \ingroup distances
 * \brief DistDistance wrapper.
 * \author Josh Klontz \cite jklontz
 */
class DefaultDistance : public Distance
{
    Q_OBJECT
    Distance *distance;

    void init()
    {
        distance = Distance::make("Dist("+file.suffix()+")");
    }

    float compare(const Template &a, const Template &b) const
    {
        return distance->compare(a, b);
    }
};

BR_REGISTER(Distance, DefaultDistance)

/*!
 * \ingroup distances
 * \brief Distances in series.
 * \author Josh Klontz \cite jklontz
 *
 * The templates are compared using each br::Distance in order.
 * If the result of the comparison with any given distance is -INT_MAX then this result is returned early.
 * Otherwise the returned result is the value of comparing the templates using the last br::Distance.
 */
class PipeDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(QList<br::Distance*> distances READ get_distances WRITE set_distances RESET reset_distances)
    BR_PROPERTY(QList<br::Distance*>, distances, QList<br::Distance*>())

    void train(const TemplateList &data)
    {
        QFutureSynchronizer<void> futures;
        foreach (br::Distance *distance, distances)
            futures.addFuture(QtConcurrent::run(distance, &Distance::train, data));
        futures.waitForFinished();
    }

    float compare(const Template &a, const Template &b) const
    {
        float result = -std::numeric_limits<float>::max();
        foreach (br::Distance *distance, distances) {
            result = distance->compare(a, b);
            if (result == -std::numeric_limits<float>::max())
                return result;
        }
        return result;
    }
};

BR_REGISTER(Distance, PipeDistance)

/*!
 * \ingroup distances
 * \brief Average distance of multiple matrices
 * \author Scott Klum \cite sklum
 */
class AverageDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    BR_PROPERTY(br::Distance*, distance, make("Dist(L2)"))

    void train(const TemplateList &src)
    {
        distance->train(src);
    }

    float compare(const Template &a, const Template &b) const
    {
        if (a.size() != b.size()) qFatal("Comparison size mismatch");

        float score = 0;
        for (int i = 0; i < a.size(); i++) {
            score += distance->compare(a[i],b[i]);
        }

        return score/(float)a.size();
    }
};

BR_REGISTER(Distance, AverageDistance)

/*!
 * \ingroup distances
 * \brief Fast 8-bit L1 distance
 * \author Josh Klontz \cite jklontz
 */
class ByteL1Distance : public Distance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        return l1(a.m().data, b.m().data, a.m().total());
    }
};

BR_REGISTER(Distance, ByteL1Distance)

/*!
 * \ingroup distances
 * \brief Fast 4-bit L1 distance
 * \author Josh Klontz \cite jklontz
 */
class HalfByteL1Distance : public Distance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        return packed_l1(a.m().data, b.m().data, a.m().total());
    }
};

BR_REGISTER(Distance, HalfByteL1Distance)

/*!
 * \ingroup distances
 * \brief Returns -log(distance(a,b)+1)
 * \author Josh Klontz \cite jklontz
 */
class NegativeLogPlusOneDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    BR_PROPERTY(br::Distance*, distance, NULL)

    void train(const TemplateList &src)
    {
        distance->train(src);
    }

    float compare(const Template &a, const Template &b) const
    {
        return -log(distance->compare(a,b)+1);
    }

    void store(QDataStream &stream) const
    {
        distance->store(stream);
    }

    void load(QDataStream &stream)
    {
        distance->load(stream);
    }
};

BR_REGISTER(Distance, NegativeLogPlusOneDistance)

/*!
 * \ingroup distances
 * \brief Returns \c true if the templates are identical, \c false otherwise.
 * \author Josh Klontz \cite jklontz
 */
class IdenticalDistance : public Distance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        const Mat &am = a.m();
        const Mat &bm = b.m();
        const size_t size = am.total() * am.elemSize();
        if (size != bm.total() * bm.elemSize()) return 0;
        for (size_t i=0; i<size; i++)
            if (am.data[i] != bm.data[i]) return 0;
        return 1;
    }
};        

BR_REGISTER(Distance, IdenticalDistance)


/*!
 * \ingroup distances
 * \brief Online distance metric to attenuate match scores across multiple frames
 * \author Brendan klare \cite bklare
 */
class OnlineDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(float alpha READ get_alpha WRITE set_alpha RESET reset_alpha STORED false)
    BR_PROPERTY(br::Distance*, distance, NULL)
    BR_PROPERTY(float, alpha, 0.1f);

    mutable QHash<QString,float> scoreHash;
    mutable QMutex mutex;

    float compare(const Template &target, const Template &query) const
    {
        float currentScore = distance->compare(target, query);

        QMutexLocker mutexLocker(&mutex);
        return scoreHash[target.file.name] = (1.0- alpha) * scoreHash[target.file.name] + alpha * currentScore;
    }
};

BR_REGISTER(Distance, OnlineDistance)

} // namespace br
#include "distance.moc"
