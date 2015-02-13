#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Convert values of key_X, key_Y, key_Width, key_Height to a rect.
 * \author Jordan Cheney \cite JordanCheney
 */
class KeyToRectTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    BR_PROPERTY(QString, key, "")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        if (src.contains(QStringList() << key + "_X" << key + "_Y" << key + "_Width" << key + "_Height"))
            dst.appendRect(QRectF(src.get<int>(key + "_X"),
                                  src.get<int>(key + "_Y"),
                                  src.get<int>(key + "_Width"),
                                  src.get<int>(key + "_Height")));

    }

};

BR_REGISTER(Transform, KeyToRectTransform)

} // namespace br

#include "keytorect.moc"
