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

#include "kmeans.moc"