#include <Eigen/Dense>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief L1 distance computed using eigen.
 * \author Josh Klontz \cite jklontz
 */
class L1Distance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const cv::Mat &a, const cv::Mat &b) const
    {
        const int size = a.rows * a.cols;
        Eigen::Map<Eigen::VectorXf> aMap((float*)a.data, size);
        Eigen::Map<Eigen::VectorXf> bMap((float*)b.data, size);
        return (aMap-bMap).cwiseAbs().sum();
    }
};

BR_REGISTER(Distance, L1Distance)

} // namespace br

#include "L1.moc"