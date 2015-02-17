#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Create matrix from metadata values.
 * \author Josh Klontz \cite jklontz
 */
class ExtractMetadataTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    BR_PROPERTY(QStringList, keys, QStringList())

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        QList<float> values;
        foreach (const QString &key, keys)
            values.append(src.file.get<float>(key));
        dst.append(OpenCVUtils::toMat(values, 1));
    }
};

BR_REGISTER(Transform, ExtractMetadataTransform)

} // namespace br

#include "metadata/extractmetadata.moc"
