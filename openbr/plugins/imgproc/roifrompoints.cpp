#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Crops the rectangular regions of interest from given points and sizes.
 * \author Austin Blanton \cite imaus10
 */
class ROIFromPtsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int width READ get_width WRITE set_width RESET reset_width STORED false)
    Q_PROPERTY(int height READ get_height WRITE set_height RESET reset_height STORED false)
    BR_PROPERTY(int, width, 1)
    BR_PROPERTY(int, height, 1)

    void project(const Template &src, Template &dst) const
    {
        foreach (const QPointF &pt, src.file.points()) {
            int x = pt.x() - (width/2);
            int y = pt.y() - (height/2);
            dst += src.m()(Rect(x, y, width, height));
        }
    }
};

BR_REGISTER(Transform, ROIFromPtsTransform)

} // namespace br

#include "roifrompoints.moc"
