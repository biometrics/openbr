#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief It's like the opposite of ExpandTransform, but not really
 * \author Charles Otto \cite caotto
 *
 * Given a set of templatelists as input, concatenate them onto a single Template
 */
class ContractTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    virtual void project(const TemplateList &src, TemplateList &dst) const
    {
        if (src.empty()) return;
        Template out;

        foreach (const Template &t, src) {
            out.merge(t);
        }
        out.file.clearRects();
        foreach (const Template &t, src) {
            if (!t.file.rects().empty())
                out.file.appendRects(t.file.rects());
        }
        dst.clear();
        dst.append(out);
    }

    virtual void project(const Template &src, Template &dst) const
    {
        qFatal("this has gone bad");
        (void) src; (void) dst;
    }
};

BR_REGISTER(Transform, ContractTransform)

} // namespace br

#include "core/contract.moc"
