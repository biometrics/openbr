#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Trim the image so the width and the height are the same size.
 * \author Josh Klontz \cite jklontz
 */
class CropSquareTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        const int newSize = min(m.rows, m.cols);
        dst = Mat(m, Rect((m.cols-newSize)/2, (m.rows-newSize)/2, newSize, newSize));
    }
};

BR_REGISTER(Transform, CropSquareTransform)

} // namespace br

#include "cropsquare.moc"
