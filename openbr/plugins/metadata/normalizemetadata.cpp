#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Normalize a floating point metadata item to be between -1 and 1. Useful for classifier labels.
 * \br_property QString inputVariable Metadata key for the item to scale. Default is empty string.
 * \br_property float min Minimum possible value for the metadata item (will be scaled to -1). Default is -1.
 * \br_property float min Maximum possible value for the metadata item (will be scaled to 1). Default is 1.
* \author Scott Klum \cite sklum
 */
class NormalizeMetadataTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(float min READ get_min WRITE set_min RESET reset_min STORED false)
    Q_PROPERTY(float max READ get_max WRITE set_max RESET reset_max STORED false)
    BR_PROPERTY(QString, inputVariable, QString())
    BR_PROPERTY(float, min, -1)
    BR_PROPERTY(float, max, 1)

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        dst.set(inputVariable,2*(dst.get<float>(inputVariable)-min)/(max-min)-1);
    }
};

BR_REGISTER(Transform, NormalizeMetadataTransform)

} // namespace br

#include "metadata/normalizemetadata.moc"
