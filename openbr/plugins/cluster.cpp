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

#include <opencv2/flann/flann.hpp>

#include "openbr_internal.h"
#include "openbr/core/common.h"
#include "openbr/core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV kmeans and flann.
 * \author Josh Klontz \cite jklontz
 */
class KMeansTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(int kTrain READ get_kTrain WRITE set_kTrain RESET reset_kTrain STORED false)
    Q_PROPERTY(int kSearch READ get_kSearch WRITE set_kSearch RESET reset_kSearch STORED false)
    BR_PROPERTY(int, kTrain, 256)
    BR_PROPERTY(int, kSearch, 1)

    Mat centers;
    mutable QScopedPointer<flann::Index> index;
    mutable QMutex mutex;

    void reindex()
    {
        index.reset(new flann::Index(centers, flann::LinearIndexParams()));
    }

    void train(const TemplateList &data)
    {
        Mat bestLabels;
        const double compactness = kmeans(OpenCVUtils::toMatByRow(data.data()), kTrain, bestLabels, TermCriteria(TermCriteria::MAX_ITER, 10, 0), 3, KMEANS_PP_CENTERS, centers);
        qDebug("KMeans compactness = %f", compactness);
        reindex();
    }

    void project(const Template &src, Template &dst) const
    {
        QMutexLocker locker(&mutex);
        Mat dists, indicies;
        index->knnSearch(src, indicies, dists, kSearch);
        dst = indicies.reshape(1, 1);
    }

    void load(QDataStream &stream)
    {
        stream >> centers;
        reindex();
    }

    void store(QDataStream &stream) const
    {
        stream << centers;
    }
};

BR_REGISTER(Transform, KMeansTransform)

/*!
 * \ingroup transforms
 * \brief K nearest subjects.
 * \author Josh Klontz \cite jklontz
 */
class KNSTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(int k READ get_k WRITE set_k RESET reset_k STORED false)
    Q_PROPERTY(br::Distance *distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    BR_PROPERTY(int, k, 1)
    BR_PROPERTY(br::Distance*, distance, NULL)

    TemplateList gallery;

    void train(const TemplateList &data)
    {
        distance->train(data);
        gallery = data;
    }

    void project(const Template &src, Template &dst) const
    {
        const QList< QPair<float, int> > sortedScores = Common::Sort(distance->compare(gallery, src), true);
        QSet<QString> subjects;
        int i = 0;
        while ((subjects.size() < k) && (i < sortedScores.size())) {
            subjects.insert(gallery[sortedScores[i].second].file.subject());
            i++;
        }
        const QStringList subjectList = subjects.toList();
        dst.file.set("KNS", subjects.size() > 1 ? "[" + subjectList.join(",") + "]" : subjectList.first());
    }

    void store(QDataStream &stream) const
    {
        stream << gallery;
    }

    void load(QDataStream &stream)
    {
        stream >> gallery;
    }
};

BR_REGISTER(Transform, KNSTransform)

} // namespace br

#include "cluster.moc"
