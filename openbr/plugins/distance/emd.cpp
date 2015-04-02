#include <openbr/plugins/openbr_internal.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

namespace br
{

/*!
 * \ingroup distances
 * \brief Fuses similarity scores across multiple matrices of compared templates
 * \author Scott Klum \cite sklum
 * \note Operation: Mean, sum, min, max are supported.
 */
class EMDDistance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        Mat sig_a(a.m().cols, 2, CV_32FC1);
        Mat sig_b(b.m().cols, 2, CV_32FC1);

        for (int i=0; i<a.m().cols; i++) {
            sig_a.at<float>(i,0) = a.m().at<float>(0,i);
            sig_a.at<float>(i,1) = i;
        }

        for (int i=0; i<b.m().cols; i++) {
            sig_b.at<float>(i,0) = b.m().at<float>(0,i);
            sig_b.at<float>(i,1) = i;
        }

        return EMD(sig_a,sig_b,CV_DIST_L2);
    }
};

BR_REGISTER(Distance, EMDDistance)

} // namespace br

#include "distance/emd.moc"
