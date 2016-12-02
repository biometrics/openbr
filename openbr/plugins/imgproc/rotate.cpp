#include <opencv2/imgproc/imgproc.hpp>
#include <openbr/plugins/openbr_internal.h>

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

         if(angle == 270 || angle == -90){
            // Rotate clockwise 270 degrees
            cv::transpose(src, dst);
            cv::flip(dst, dst, 0);
        }else if(angle == 180 || angle == -180){
            // Rotate clockwise 180 degrees
            cv::flip(src, dst, -1);
        }else if(angle == 90 || angle == -270){
            // Rotate clockwise 90 degrees
            cv::transpose(src, dst);
            cv::flip(dst, dst, 1);
        }else {
            dst = src;
        }
    }
};

BR_REGISTER(Transform, RotateTransform)

} // namespace br

#include "imgproc/rotate.moc"

