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
#include <openbr/core/common.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

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

    void store(QDataStream &stream) const
    {
        stream << centers;
    }
};

BR_REGISTER(Transform, RandomCentroidsTransform)

} //namespace br

#include "cluster/randomcentroids.moc"
