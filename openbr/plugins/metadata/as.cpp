#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Change the br::Template::file extension
 * \author Josh Klontz \cite jklontz
 */
class AsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString extension READ get_extension WRITE set_extension RESET reset_extension STORED false)
    BR_PROPERTY(QString, extension, "")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        dst.name = dst.name.left(dst.name.lastIndexOf('.')+1) + extension;
    }
};

BR_REGISTER(Transform, AsTransform)

} // namespace br

#include "metadata/as.moc"
