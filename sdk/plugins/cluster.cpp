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

#include <openbr_plugin.h>
#include <opencv2/flann/flann.hpp>

#include "core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV kmeans
 * \author Josh Klontz \cite jklontz
 */
class KMeansTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(int k READ get_k WRITE set_k RESET reset_k)
    BR_PROPERTY(int, k, 1)

    Mat centers;
    QSharedPointer<flann::Index> index;
    QSharedPointer<QMutex> indexLock;

    void reindex()
    {
        index = QSharedPointer<flann::Index>(new flann::Index(centers, flann::LinearIndexParams()));
        indexLock = QSharedPointer<QMutex>(new QMutex());
    }

    void train(const TemplateList &data)
    {
        Mat bestLabels;
        const double compactness = kmeans(OpenCVUtils::toMatByRow(data.data()), k, bestLabels, TermCriteria(TermCriteria::MAX_ITER, 10, 0), 3, KMEANS_PP_CENTERS, centers);
        reindex();
        qDebug("KMeans compactness = %f", compactness);
    }

    void project(const Template &src, Template &dst) const
    {
        Mat dists;
        indexLock->lock();
        index->knnSearch(src, dst, dists, 1);
        indexLock->unlock();
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

}

#include "cluster.moc"
