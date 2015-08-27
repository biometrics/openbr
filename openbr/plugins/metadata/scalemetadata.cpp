#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Scales a floating point metadata item by factor
 * \br_property QString inputVariable Metadata key for the item to scale. Default is empty string.
 * \br_property float factor Floating point factor to scale the item. Default is 1.
 * \author Scott Klum \cite sklum
 */
class ScaleMetadataTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(float factor READ get_factor WRITE set_factor RESET reset_factor STORED false)
    BR_PROPERTY(QString, inputVariable, QString())
    BR_PROPERTY(float, factor, 1)

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        dst.set(inputVariable,dst.get<float>(inputVariable)*factor);
    }
};

BR_REGISTER(Transform, ScaleMetadataTransform)

} // namespace br

#include "metadata/scalemetadata.moc"
