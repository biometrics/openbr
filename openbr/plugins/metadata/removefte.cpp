#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \author Brendan Klare \cite bklare
 * \brief Remove any templates that failed to enroll (FTE).
 * 	Important note: this will not work without the global enrollAll being true
 */
class RemoveFTETransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &, Template &) const
    {
        qFatal("Not supported in RemoveFTE.");
    }

    void project(const TemplateList &src, TemplateList &dst) const  
    {
        for (int i = 0; i < src.size(); i++) 
            if (!src[i].file.fte)
                dst.append(src[i]);
    }

    void docs(int indent) const
    {
        print_doc("RemoveFTE(): Remove any templates marked FTE", indent);
    }
};
BR_REGISTER(Transform, RemoveFTETransform)

} // namespace br

#include "imgproc/removefte.moc"

