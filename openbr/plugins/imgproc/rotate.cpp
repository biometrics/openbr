#include <opencv2/imgproc/imgproc.hpp>
#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Rotates an image in 90 degree intervals.
 * \author Keyur Patel \cite patel
 */
class RotateTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(int angle READ get_angle WRITE set_angle RESET reset_angle STORED false)
    BR_PROPERTY(int, angle, -90)

    void project(const Template &src, Template &dst) const {
        OpenCVUtils::rotate(src, dst, angle);
    }
};

BR_REGISTER(Transform, RotateTransform)

} // namespace br

#include "imgproc/rotate.moc"

