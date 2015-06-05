#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's adaptive thresholding.
 * \br_link http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html
 * \author Scott Klum \cite sklum
 */
class AdaptiveThresholdTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_ENUMS(Method)
    Q_ENUMS(Type)
    Q_PROPERTY(int maxValue READ get_maxValue WRITE set_maxValue RESET reset_maxValue STORED false)
    Q_PROPERTY(Method method READ get_method WRITE set_method RESET reset_method STORED false)
    Q_PROPERTY(Type type READ get_type WRITE set_type RESET reset_type STORED false)
    Q_PROPERTY(int blockSize READ get_blockSize WRITE set_blockSize RESET reset_blockSize STORED false)
    Q_PROPERTY(int C READ get_C WRITE set_C RESET reset_C STORED false)

    public:
    enum Method { Mean = ADAPTIVE_THRESH_MEAN_C,
                  Gaussian = ADAPTIVE_THRESH_GAUSSIAN_C };

    enum Type { Binary = THRESH_BINARY,
                BinaryInv = THRESH_BINARY_INV };

private:
    BR_PROPERTY(int, maxValue, 255)
    BR_PROPERTY(Method, method, Mean)
    BR_PROPERTY(Type, type, Binary)
    BR_PROPERTY(int, blockSize, 3)
    BR_PROPERTY(int, C, 0)

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        Mat mask;
        adaptiveThreshold(src, mask, maxValue, method, type, blockSize, C);

        dst.file.set("Mask",QVariant::fromValue(mask));
    }
};

BR_REGISTER(Transform, AdaptiveThresholdTransform)

} // namespace br

#include "imgproc/adaptivethreshold.moc"
