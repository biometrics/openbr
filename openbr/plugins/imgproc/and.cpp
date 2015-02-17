#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Logical AND of two matrices.
 * \author Josh Klontz \cite jklontz
 */
class AndTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        dst.m() = src.first();
        for (int i=1; i<src.size(); i++)
            cv::bitwise_and(src[i], dst, dst);
    }
};

BR_REGISTER(Transform, AndTransform)

} // namespace br

#include "imgproc/and.moc"
