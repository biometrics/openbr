#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Crops the width and height of a template's rects by input width and height factors.
 * \author Scott Klum \cite sklum
 */
class CropRectTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    Q_PROPERTY(QString widthCrop READ get_widthCrop WRITE set_widthCrop RESET reset_widthCrop STORED false)
    Q_PROPERTY(QString heightCrop READ get_heightCrop WRITE set_heightCrop RESET reset_heightCrop STORED false)
    BR_PROPERTY(QString, widthCrop, QString())
    BR_PROPERTY(QString, heightCrop, QString())

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        QList<QRectF> rects = src.rects();
        for (int i=0;i < rects.size(); i++) {
            rects[i].setX(rects[i].x() + rects[i].width() * QtUtils::toPoint(widthCrop).x());
            rects[i].setY(rects[i].y() + rects[i].height() * QtUtils::toPoint(heightCrop).x());
            rects[i].setWidth(rects[i].width() * (1-QtUtils::toPoint(widthCrop).y()));
            rects[i].setHeight(rects[i].height() * (1-QtUtils::toPoint(heightCrop).y()));
        }
        dst.setRects(rects);
    }
};

BR_REGISTER(Transform, CropRectTransform)

} // namespace br

#include "metadata/croprect.moc"
