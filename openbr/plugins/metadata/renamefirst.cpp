#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Rename first found metadata key
 * \author Josh Klontz \cite jklontz
 */
class RenameFirstTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList find READ get_find WRITE set_find RESET reset_find STORED false)
    Q_PROPERTY(QString replace READ get_replace WRITE set_replace RESET reset_replace STORED false)
    BR_PROPERTY(QStringList, find, QStringList())
    BR_PROPERTY(QString, replace, "")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        foreach (const QString &key, find)
            if (dst.localKeys().contains(key)) {
                dst.set(replace, dst.value(key));
                dst.remove(key);
                break;
            }
    }
};

BR_REGISTER(Transform, RenameFirstTransform)

} // namespace br

#include "metadata/renamefirst.moc"
