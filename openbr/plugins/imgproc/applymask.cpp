#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Applies a mask from the metadata.
 * \author Austin Blanton \cite imaus10
 */
class ApplyMaskTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        if (src.file.contains("Mask"))
            src.m().copyTo(dst, src.file.get<Mat>("Mask"));
        else
            dst = src;
    }
};

BR_REGISTER(Transform, ApplyMaskTransform)

} // namespace br

#include "imgproc/applymask.moc"
