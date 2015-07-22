#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Convert a rect to X, Y, Width, and Height. Handy for saving universal templates.
 * \author Austin Blanton \cite imaus10
 */
class RectToKeysTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    BR_PROPERTY(QString, key, "")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        if (src.contains(key)) {
            QRectF r = src.get<QRectF>(key);
            dst.set("Height", r.height());
            dst.set("Width", r.width());
            dst.set("X", r.left());
            dst.set("Y", r.bottom());
        }
    }

};

BR_REGISTER(Transform, RectToKeysTransform)

} // namespace br

#include "metadata/recttokeys.moc"
