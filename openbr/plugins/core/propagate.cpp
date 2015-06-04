#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Ensures that a template will be propogated.
 * \author Scott Klum \cite sklum
 */
class PropagateTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(br::Transform *transform READ get_transform WRITE set_transform RESET reset_transform STORED true)
    BR_PROPERTY(br::Transform *, transform, NULL)

    void project(const Template &src, Template &dst) const
    {
        transform->project(src,dst);
        if (dst.isEmpty())
            dst = src;
    }
};

BR_REGISTER(Transform, PropagateTransform)

} // namespace br

#include "core/propagate.moc"
