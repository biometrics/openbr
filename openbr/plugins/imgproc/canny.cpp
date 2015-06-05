#include "opencv2/imgproc/imgproc.hpp"

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wrapper to OpenCV Canny edge detector
 * \br_link http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
 * \author Scott Klum \cite sklum
 */
class CannyTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(double threshold READ get_threshold WRITE set_threshold RESET reset_threshold STORED false)
    Q_PROPERTY(double aperatureSize READ get_aperatureSize WRITE set_aperatureSize RESET reset_aperatureSize STORED false)
    Q_PROPERTY(bool L2Gradient READ get_L2Gradient WRITE set_L2Gradient RESET reset_L2Gradient STORED false)
    BR_PROPERTY(double, threshold, 5)
    BR_PROPERTY(double, aperatureSize, 3)
    BR_PROPERTY(bool, L2Gradient, true)

    void project(const Template &src, Template &dst) const
    {
        Canny(src,dst, threshold, 3*threshold, aperatureSize, L2Gradient);
    }
};

BR_REGISTER(Transform, CannyTransform)

} // namespace br

#include "imgproc/canny.moc"
