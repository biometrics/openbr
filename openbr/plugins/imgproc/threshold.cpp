#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's adaptive thresholding.
 * \author Scott Klum \cite sklum
 */
class ThresholdTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_ENUMS(Method)
    Q_ENUMS(Type)
    Q_PROPERTY(int thresh READ get_thresh WRITE set_thresh RESET reset_thresh STORED false)
    Q_PROPERTY(int maxValue READ get_maxValue WRITE set_maxValue RESET reset_maxValue STORED false)

    public:
    BR_PROPERTY(int, thresh, 0)
    BR_PROPERTY(int, maxValue, 255)


    void project(const Template &src, Template &dst) const
    {
        dst = src;

        Mat mask;
        threshold(src, mask, thresh, maxValue, THRESH_BINARY+THRESH_OTSU);

        dst.file.set("Mask",QVariant::fromValue(mask));
    }
};

BR_REGISTER(Transform, ThresholdTransform)

} // namespace br

#include "imgproc/threshold.moc"
