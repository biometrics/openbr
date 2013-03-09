#include <opencv2/imgproc/imgproc.hpp>
#include <openbr_plugin.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Computes integral image.
 * \author Josh Klontz \cite jklontz
 */
class IntegralTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        cv::integral(src, dst);
    }
};

BR_REGISTER(Transform, IntegralTransform)

/*!
 * \ingroup transforms
 * \brief Computes magnitude and/or angle of image.
 * \author Josh Klontz \cite jklontz
 */
class GradientTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Channel)
    Q_PROPERTY(Channel channel READ get_channel WRITE set_channel RESET reset_channel STORED false)

public:
    enum Channel { Magnitude, Angle, MagnitudeAndAngle };

private:
    BR_PROPERTY(Channel, channel, Angle)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().type() != CV_8UC1) qFatal("Requires CV_8UC1 input.");
        cv::Mat dx, dy, magnitude, angle;
        cv::Sobel(src, dx, CV_32F, 1, 0);
        cv::Sobel(src, dy, CV_32F, 0, 1);
        cv::cartToPolar(dx, dy, magnitude, angle, true);
        if ((channel == Magnitude) || (channel == MagnitudeAndAngle))
            dst.append(magnitude);
        if ((channel == Angle) || (channel == MagnitudeAndAngle))
            dst.append(angle);
    }
};

BR_REGISTER(Transform, GradientTransform)

} // namespace br

#include "integral.moc"
