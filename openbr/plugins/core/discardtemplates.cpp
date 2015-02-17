#include <openbr/plugins/openbr_internal.h>

namespace br
{

class DiscardTemplatesTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        (void) src; (void) dst;
        qFatal("Incorrect project called on DiscardTemplatesTransform");
    }
    void project(const TemplateList &src, TemplateList &dst) const
    {
        (void) src;
        dst.clear();
    }
};

BR_REGISTER(Transform, DiscardTemplatesTransform)

} // namespace br

#include "core/discardtemplates.moc"
