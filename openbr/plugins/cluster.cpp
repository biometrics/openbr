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

    void store(QDataStream &stream, bool force) const
    {
        stream << centers;
    }
};

BR_REGISTER(Transform, KMeansTransform)

/*!
 * \ingroup transforms
 * \brief K nearest neighbors classifier.
 * \author Josh Klontz \cite jklontz
 */
class KNNTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(int k READ get_k WRITE set_k RESET reset_k STORED false)
    Q_PROPERTY(br::Distance *distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(bool weighted READ get_weighted WRITE set_weighted RESET reset_weighted STORED false)
    Q_PROPERTY(int numSubjects READ get_numSubjects WRITE set_numSubjects RESET reset_numSubjects STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(QString outputVariable READ get_outputVariable WRITE set_outputVariable RESET reset_outputVariable STORED false)
    Q_PROPERTY(QString galleryName READ get_galleryName WRITE set_galleryName RESET reset_galleryName STORED false)
    BR_PROPERTY(int, k, 1)
    BR_PROPERTY(br::Distance*, distance, NULL)
    BR_PROPERTY(bool, weighted, false)
    BR_PROPERTY(int, numSubjects, 1)
    BR_PROPERTY(QString, inputVariable, "Label")
    BR_PROPERTY(QString, outputVariable, "KNN")
    BR_PROPERTY(QString, galleryName, "")

    TemplateList gallery;

    void train(const TemplateList &data)
    {
        distance->train(data);
        gallery = data;
    }

    void project(const Template &src, Template &dst) const
    {
        QList< QPair<float, int> > sortedScores = Common::Sort(distance->compare(gallery, src), true);

        QStringList subjects;
        for (int i=0; i<numSubjects; i++) {
            QHash<QString, float> votes;
            const int max = (k < 1) ? sortedScores.size() : std::min(k, sortedScores.size());
            for (int j=0; j<max; j++)
                votes[gallery[sortedScores[j].second].file.get<QString>(inputVariable)] += (weighted ? sortedScores[j].first : 1);
            subjects.append(votes.keys()[votes.values().indexOf(Common::Max(votes.values()))]);

            // Remove subject from consideration
            if (subjects.size() < numSubjects)
                for (int j=sortedScores.size()-1; j>=0; j--)
                    if (gallery[sortedScores[j].second].file.get<QString>(inputVariable) == subjects.last())
                        sortedScores.removeAt(j);
        }

        dst.file.set(outputVariable, subjects.size() > 1 ? "[" + subjects.join(",") + "]" : subjects.first());
        dst.file.set("Nearest", gallery[sortedScores[0].second].file.name);
    }

    void store(QDataStream &stream, bool force) const
    {
        stream << gallery;
    }

    void load(QDataStream &stream)
    {
        stream >> gallery;
    }

    void init()
    {
        if (!galleryName.isEmpty())
            gallery = TemplateList::fromGallery(galleryName);
    }
};

BR_REGISTER(Transform, KNNTransform)

/*!
 * \ingroup transforms
 * \brief Chooses k random points to be centroids.
 * \author Austin Blanton \cite imaus10
 * \see KMeansTransform
 */
class RandomCentroidsTransform : public Transform
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
        Mat flat = OpenCVUtils::toMatByRow(data.data());
        QList<int> sample = Common::RandSample(kTrain, flat.rows, 0, true);
        foreach (const int &idx, sample)
            centers.push_back(flat.row(idx));
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

    void store(QDataStream &stream, bool force) const
    {
        stream << centers;
    }
};

BR_REGISTER(Transform, RandomCentroidsTransform)

} // namespace br

#include "cluster.moc"
