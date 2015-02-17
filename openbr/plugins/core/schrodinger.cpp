#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Generates two templates, one of which is passed through a transform and the other
 *        is not. No cats were harmed in the making of this transform.
 * \author Scott Klum \cite sklum
 */
class SchrodingerTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform)
    BR_PROPERTY(br::Transform*, transform, NULL)

public:
    void train(const TemplateList &data)
    {
        transform->train(data);
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach(const Template &t, src) {
            dst.append(t);
            Template u;
            transform->project(t,u);
            dst.append(u);
        }
    }

    void project(const Template &src, Template &dst) const {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

};
BR_REGISTER(Transform, SchrodingerTransform)

} // namespace br

#include "core/schrodinger.moc"
