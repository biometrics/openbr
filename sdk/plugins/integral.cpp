#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include <openbr_plugin.h>

#include "core/opencvutils.h"

using namespace cv;

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
        integral(src, dst);
    }
};

BR_REGISTER(Transform, IntegralTransform)

/*!
 * \ingroup transforms
 * \brief Sliding window feature extraction from a multi-channel intergral image.
 * \author Josh Klontz \cite jklontz
 */
class IntegralSamplerTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int scales READ get_scales WRITE set_scales RESET reset_scales STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(float stepFactor READ get_stepFactor WRITE set_stepFactor RESET reset_stepFactor STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    BR_PROPERTY(int, scales, 4)
    BR_PROPERTY(float, scaleFactor, 1.5)
    BR_PROPERTY(float, stepFactor, 0.25)
    BR_PROPERTY(int, minSize, 8)

    void project(const Template &src, Template &dst) const
    {
        typedef Eigen::Map< const Eigen::Matrix<qint32,Eigen::Dynamic,1> > InputDescriptor;
        typedef Eigen::Map< Eigen::Matrix<float,Eigen::Dynamic,1> > OutputDescriptor;
        const Mat &m = src.m();
        if (m.depth() != CV_32S) qFatal("Expected CV_32S matrix depth.");
        const int channels = m.channels();
        const int rowStep = channels * m.cols;

        int descriptors = 0;
        int currentSize = min(m.rows, m.cols)-1;
        for (int scale=0; scale<scales; scale++) {
            descriptors += int(1+(m.rows-currentSize)/(currentSize*stepFactor)) *
                           int(1+(m.cols-currentSize)/(currentSize*stepFactor));
            currentSize /= scaleFactor;
            if (currentSize < minSize)
                break;
        }
        Mat n(descriptors, channels, CV_32FC1);

        const qint32 *dataIn = (qint32*)m.data;
        float *dataOut = (float*)n.data;
        currentSize = min(m.rows, m.cols)-1;
        int index = 0;
        for (int scale=0; scale<scales; scale++) {
            const int currentStep = currentSize * stepFactor;
            for (int i=currentSize; i<m.rows; i+=currentStep) {
                for (int j=currentSize; j<m.cols; j+=currentStep) {
                    InputDescriptor a(dataIn+((i-currentSize)*rowStep+(j-currentSize)*channels), channels, 1);
                    InputDescriptor b(dataIn+((i-currentSize)*rowStep+ j             *channels), channels, 1);
                    InputDescriptor c(dataIn+(i              *rowStep+(j-currentSize)*channels), channels, 1);
                    InputDescriptor d(dataIn+(i              *rowStep+ j             *channels), channels, 1);
                    OutputDescriptor y(dataOut+(index*channels), channels, 1);
                    y = (d-b-c+a).cast<float>()/(currentSize*currentSize);
                    index++;
                }
            }
            currentSize /= scaleFactor;
            if (currentSize < minSize)
                break;
        }
        dst.m() = n;
    }
};

BR_REGISTER(Transform, IntegralSamplerTransform)

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
        Mat dx, dy, magnitude, angle;
        Sobel(src, dx, CV_32F, 1, 0);
        Sobel(src, dy, CV_32F, 0, 1);
        cartToPolar(dx, dy, magnitude, angle, true);
        if ((channel == Magnitude) || (channel == MagnitudeAndAngle))
            dst.append(magnitude);
        if ((channel == Angle) || (channel == MagnitudeAndAngle))
            dst.append(angle);
    }
};

BR_REGISTER(Transform, GradientTransform)

} // namespace br

#include "integral.moc"
