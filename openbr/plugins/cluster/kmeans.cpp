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

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV kmeans and flann.
 * \author Josh Klontz \cite jklontz
 * \br_property int kTrain The number of random centroids to make at train time. Default is 256.
 * \br_property int kSearch The number of nearest neighbors to search for at runtime. Default is 1.
 * \br_link http://docs.opencv.org/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html
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

} // namespace br

#include "cluster/kmeans.moc"
