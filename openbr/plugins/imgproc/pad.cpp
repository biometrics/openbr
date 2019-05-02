#include <opencv2/imgproc/imgproc.hpp>
#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Pads an image.
 * \author Scott Klum \cite sklum
 */
class PadTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Border)
    Q_PROPERTY(Border border READ get_border WRITE set_border RESET reset_border STORED false)
    Q_PROPERTY(float percent READ get_percent WRITE set_percent RESET reset_percent STORED false)
    Q_PROPERTY(int value READ get_value WRITE set_value RESET reset_value STORED false)

public:
    /*!< */
    enum Border { Replicate = BORDER_REPLICATE,
                  Reflect = BORDER_REFLECT_101,
                  Constant = BORDER_CONSTANT};

private:
    BR_PROPERTY(Border, border, Replicate)
    BR_PROPERTY(float, percent, .1)
    BR_PROPERTY(float, value, 0)

    void project(const Template &src, Template &dst) const
    {
        int top, bottom, left, right;
        top = percent*src.m().rows; bottom = percent*src.m().rows;
        left = percent*src.m().cols; right = percent*src.m().cols;
        OpenCVUtils::pad(src, dst, true, QMargins(left, top, right, bottom), true, true, border, value);
    }
};

BR_REGISTER(Transform, PadTransform)

} // namespace br

#include "imgproc/pad.moc"
