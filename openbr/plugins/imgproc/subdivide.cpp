#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Divide the matrix into 4 smaller matricies of equal size.
 * \author Josh Klontz \cite jklontz
 */
class SubdivideTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        const int subrows = m.rows/2;
        const int subcolumns = m.cols/2;
        dst.append(Mat(m,Rect(0,          0, subcolumns, subrows)).clone());
        dst.append(Mat(m,Rect(subcolumns, 0, subcolumns, subrows)).clone());
        dst.append(Mat(m,Rect(0,          subrows, subcolumns, subrows)).clone());
        dst.append(Mat(m,Rect(subcolumns, subrows, subcolumns, subrows)).clone());
    }
};

BR_REGISTER(Transform, SubdivideTransform)

} // namespace br

#include "imgproc/subdivide.moc"
