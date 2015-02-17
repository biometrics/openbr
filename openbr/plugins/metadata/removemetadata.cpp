#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Removes a metadata field from all templates
 * \author Brendan Klare \cite bklare
 */
class RemoveMetadataTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString attributeName READ get_attributeName WRITE set_attributeName RESET reset_attributeName STORED false)
    BR_PROPERTY(QString, attributeName, "None")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        if (dst.contains(attributeName))
            dst.remove(attributeName);
    }
};

BR_REGISTER(Transform, RemoveMetadataTransform)

} // namespace br

#include "metadata/removemetadata.moc"
