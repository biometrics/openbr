#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \author Brendan Klare \cite bklare
 * \brief Remove any templates that failed to enroll (FTE)
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
};
BR_REGISTER(Transform, RemoveFTETransform)

} // namespace br

#include "imgproc/removefte.moc"

