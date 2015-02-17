#include <openbr/plugins/openbr_internal.h>

namespace br
{

class TransposeTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.m() = src.m().t();
    }
};

BR_REGISTER(Transform, TransposeTransform)

} // namespace br

#include "imgproc/transpose.moc"
