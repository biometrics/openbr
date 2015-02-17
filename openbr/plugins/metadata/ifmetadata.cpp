#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Clear templates without the required metadata.
 * \author Josh Klontz \cite jklontz
 */
class IfMetadataTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    Q_PROPERTY(QString value READ get_value WRITE set_value RESET reset_value STORED false)
    BR_PROPERTY(QString, key, "")
    BR_PROPERTY(QString, value, "")

    void projectMetadata(const File &src, File &dst) const
    {
        if (src.get<QString>(key, "") == value)
            dst = src;
    }
};

BR_REGISTER(Transform, IfMetadataTransform)

} // namespace br

#include "metadata/ifmetadata.moc"
