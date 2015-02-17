#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Remove a name from a point/rect
 * \author Scott Klum \cite sklum
 */
class AnonymizeLandmarksTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList names READ get_names WRITE set_names RESET reset_names STORED false)
    BR_PROPERTY(QStringList, names, QStringList())

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        foreach (const QString &name, names) {
            if (src.contains(name)) {
                QVariant variant = src.value(name);
                if (variant.canConvert(QMetaType::QPointF)) {
                    dst.appendPoint(variant.toPointF());
                } else if (variant.canConvert(QMetaType::QRectF)) {
                    dst.appendRect(variant.toRectF());
                } else {
                    qFatal("Cannot convert landmark to point or rect.");
                }
            }
        }
    }
};

BR_REGISTER(Transform, AnonymizeLandmarksTransform)

} // namespace br

#include "metadata/anonymizelandmarks.moc"
