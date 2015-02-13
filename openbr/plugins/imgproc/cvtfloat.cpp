#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Convert to floating point format.
 * \author Josh Klontz \cite jklontz
 */
class CvtFloatTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        src.m().convertTo(dst, CV_32F);
    }
};

BR_REGISTER(Transform, CvtFloatTransform)

} // namespace br

#include "cvtfloat.moc"
