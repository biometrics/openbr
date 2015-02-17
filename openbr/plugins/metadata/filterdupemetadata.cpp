#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Removes duplicate templates based on a unique metadata key
 * \author Austin Blanton \cite imaus10
 */
class FilterDupeMetadataTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    BR_PROPERTY(QString, key, "TemplateID")

    QSet<QString> excluded;

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        foreach (const Template &t, src) {
            QString id = t.file.get<QString>(key);
            if (!excluded.contains(id)) {
                dst.append(t);
                excluded.insert(id);
            }
        }
    }

public:
    FilterDupeMetadataTransform() : TimeVaryingTransform(false,false) {}
};

BR_REGISTER(Transform, FilterDupeMetadataTransform)

} // namespace br

#include "metadata/filterdupemetadata.moc"
