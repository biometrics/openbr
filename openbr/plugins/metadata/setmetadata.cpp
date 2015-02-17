#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Sets the metadata key/value pair.
 * \author Josh Klontz \cite jklontz
 */
class SetMetadataTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    Q_PROPERTY(QString value READ get_value WRITE set_value RESET reset_value STORED false)
    BR_PROPERTY(QString, key, "")
    BR_PROPERTY(QString, value, "")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        dst.set(key, value);
    }
};

BR_REGISTER(Transform, SetMetadataTransform)

} // namespace br

#include "metadata/setmetadata.moc"
