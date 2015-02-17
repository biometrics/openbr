#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Retains only the values for the keys listed, to reduce template size
 * \author Scott Klum \cite sklum
 */
class KeepMetadataTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    BR_PROPERTY(QStringList, keys, QStringList())

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        foreach (const QString& localKey, dst.localKeys())
            if (!keys.contains(localKey))
                dst.remove(localKey);
    }
};

BR_REGISTER(Transform, KeepMetadataTransform)

} // namespace br

#include "metadata/keepmetadata.moc"
