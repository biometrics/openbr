#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Scales using the given factor
 * \author Scott Klum \cite sklum
 */
class ScaleTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    BR_PROPERTY(float, scaleFactor, 1.)

    void project(const Template &src, Template &dst) const
    {
         resize(src, dst, Size(src.m().cols*scaleFactor,src.m().rows*scaleFactor));
    }
};

BR_REGISTER(Transform, ScaleTransform)

} // namespace br

#include "imgproc/scale.moc"
