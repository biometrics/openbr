#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Enforce a multiple of \em n columns.
 * \author Josh Klontz \cite jklontz
 */
class DivTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    BR_PROPERTY(int, n, 1)

    void project(const Template &src, Template &dst) const
    {
        dst = Mat(src, Rect(0,0,n*(src.m().cols/n),src.m().rows));
    }
};

BR_REGISTER(Transform, DivTransform)

} // namespace br

#include "imgproc/div.moc"
