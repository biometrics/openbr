#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Crops about the specified region of interest.
 * \author Josh Klontz \cite jklontz
 */
class CropTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int x READ get_x WRITE set_x RESET reset_x STORED false)
    Q_PROPERTY(int y READ get_y WRITE set_y RESET reset_y STORED false)
    Q_PROPERTY(int width READ get_width WRITE set_width RESET reset_width STORED false)
    Q_PROPERTY(int height READ get_height WRITE set_height RESET reset_height STORED false)
    BR_PROPERTY(int, x, 0)
    BR_PROPERTY(int, y, 0)
    BR_PROPERTY(int, width, -1)
    BR_PROPERTY(int, height, -1)

    void project(const Template &src, Template &dst) const
    {
        dst = Mat(src, Rect(x, y, width < 1 ? src.m().cols-x-abs(width) : width, height < 1 ? src.m().rows-y-abs(height) : height));
    }
};

BR_REGISTER(Transform, CropTransform)

} // namespace br

#include "crop.moc"
