#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Statistics
 * \author Josh Klontz \cite jklontz
 */
class MatStatsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Statistic)
    Q_PROPERTY(Statistic statistic READ get_statistic WRITE set_statistic RESET reset_statistic STORED false)

public:
    /*!
     * \brief Available statistics
     */
    enum Statistic { Min, Max, Mean, StdDev };

private:
    BR_PROPERTY(Statistic, statistic, Mean)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() != 1)
            qFatal("Expected 1 channel matrix.");
        Mat m(1, 1, CV_32FC1);
        if ((statistic == Min) || (statistic == Max)) {
            double min, max;
            minMaxLoc(src, &min, &max);
            m.at<float>(0,0) = (statistic == Min ? min : max);
        } else {
            Scalar mean, stddev;
            meanStdDev(src, mean, stddev);
            m.at<float>(0,0) = (statistic == Mean ? mean[0] : stddev[0]);
        }
        dst = m;
    }
};

BR_REGISTER(Transform, MatStatsTransform)

} // namespace br

#include "imgproc/matstats.moc"
