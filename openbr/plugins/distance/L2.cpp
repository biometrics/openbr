#include <Eigen/Dense>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief L2 distance computed using eigen.
 * \author Josh Klontz \cite jklontz
 */
class L2Distance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const cv::Mat &a, const cv::Mat &b) const
    {
        const int size = a.rows * a.cols;
        Eigen::Map<Eigen::VectorXf> aMap((float*)a.data, size);
        Eigen::Map<Eigen::VectorXf> bMap((float*)b.data, size);
        return (aMap-bMap).squaredNorm();
    }
};

BR_REGISTER(Distance, L2Distance)

} // namespace br

#include "distance/L2.moc"
