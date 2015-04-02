#include <openbr/plugins/openbr_internal.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

namespace br
{

/*!
 * \ingroup distances
 * \brief Computes Earth Mover's Distance
 * \author Scott Klum \cite sklum
 * \brief https://www.cs.duke.edu/~tomasi/papers/rubner/rubnerTr98.pdf
 */
class EMDDistance : public UntrainableDistance
{
    Q_OBJECT

    Q_ENUMS(Metric)
    Q_PROPERTY(Metric metric READ get_metric WRITE set_metric RESET reset_metric STORED false)

public:
    enum Metric { L1 = CV_DIST_L1,
                  L2 = CV_DIST_L2,
                  C = CV_DIST_C };

private:
    BR_PROPERTY(Metric, metric, L2)

    float compare(const Template &a, const Template &b) const
    {
        const int dims_a = a.m().rows > 1 ? 3 : 2;
        const int dims_b = b.m().rows > 1 ? 3 : 2;

        Mat sig_a(a.m().cols, dims_a, CV_32FC1);
        Mat sig_b(b.m().cols, dims_b, CV_32FC1);

        for (int i=0; i<a.m().rows; i++) {
            for (int j=0; j<a.m().cols; j++) {
                sig_a.at<float>(i*a.m().cols+j,0) = a.m().at<float>(i,j);
                sig_a.at<float>(i*a.m().cols+j,1) = j;
                if (dims_a == 3) sig_a.at<float>(i*a.m().cols+j,2) = i;
            }
        }

        for (int i=0; i<b.m().rows; i++) {
            for (int j=0; j<b.m().cols; j++) {
                sig_b.at<float>(i*b.m().cols+j,0) = b.m().at<float>(i,j);
                sig_b.at<float>(i*b.m().cols+j,1) = j;
                if (dims_b == 3) sig_a.at<float>(i*b.m().cols+j,2) = i;
            }
        }

        return EMD(sig_a,sig_b,metric);
    }
};

BR_REGISTER(Distance, EMDDistance)

} // namespace br

#include "distance/emd.moc"
