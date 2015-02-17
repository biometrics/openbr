#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Take the absolute difference of two matrices.
 * \author Josh Klontz \cite jklontz
 */
class AbsDiffTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        if (src.size() != 2) qFatal("Expected exactly two source images, got %d.", src.size());
        dst.file = src.file;
        cv::absdiff(src[0], src[1], dst);
    }
};

BR_REGISTER(Transform, AbsDiffTransform)

} // namespace br

#include "imgproc/absdiff.moc"
