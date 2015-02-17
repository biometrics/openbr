#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Fill 0 pixels with the mean of non-0 pixels.
 * \author Josh Klontz \cite jklontz
 */
class MeanFillTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src.m().clone();
        dst.m().setTo(mean(dst, dst.m()!=0), dst.m()==0);
    }
};

BR_REGISTER(Transform, MeanFillTransform)

} // namespace br

#include "imgproc/meanfill.moc"
