#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Add any ground truth to the template using the file's base name.
 * \author Josh Klontz \cite jklontz
 */
class GroundTruthTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString groundTruth READ get_groundTruth WRITE set_groundTruth RESET reset_groundTruth STORED false)
    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    BR_PROPERTY(QString, groundTruth, "")
    BR_PROPERTY(QStringList, keys, QStringList())

    QMap<QString,File> files;

    void init()
    {
        foreach (const File &file, TemplateList::fromGallery(groundTruth).files())
            files.insert(file.baseName(), file);
    }

    void projectMetadata(const File &src, File &dst) const
    {
        (void) src;
        foreach(const QString &key, keys)
            dst.set(key,files[dst.baseName()].value(key));
    }
};

BR_REGISTER(Transform, GroundTruthTransform)

} // namespace br

#include "metadata/groundtruth.moc"
