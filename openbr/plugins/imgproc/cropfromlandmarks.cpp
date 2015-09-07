#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Crops around the landmarks numbers provided.
 * \author Brendan Klare \cite bklare
 * \param padding Percentage of height and width to pad the image.
 */
class CropFromLandmarksTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    Q_PROPERTY(float paddingHorizontal READ get_paddingHorizontal WRITE set_paddingHorizontal RESET reset_paddingHorizontal STORED false)
    Q_PROPERTY(float paddingVertical READ get_paddingVertical WRITE set_paddingVertical RESET reset_paddingVertical STORED false)
    BR_PROPERTY(QList<int>, indices, QList<int>())
    BR_PROPERTY(float, paddingHorizontal, .1)
    BR_PROPERTY(float, paddingVertical, .1)

    void project(const Template &src, Template &dst) const
    {
        int minX = src.m().cols - 1,
            maxX = 1,
            minY = src.m().rows - 1,
            maxY = 1;

        for (int i = 0; i <indices.size(); i++) {
            if (minX > src.file.points()[indices[i]].x())
                minX = src.file.points()[indices[i]].x();
            if (minY > src.file.points()[indices[i]].y())
                minY = src.file.points()[indices[i]].y();
            if (maxX < src.file.points()[indices[i]].x())
                maxX = src.file.points()[indices[i]].x();
            if (maxY < src.file.points()[indices[i]].y())
                maxY = src.file.points()[indices[i]].y();
        }

        int padW = qRound((maxX - minX) * (paddingHorizontal / 2));
        int padH = qRound((maxY - minY) * (paddingVertical / 2));

        QRectF rect(minX - padW, minY - padH, (maxX - minX + 1) + padW * 2, (maxY - minY + 1) + padH * 2);
        if (rect.x() < 0) rect.setX(0);
        if (rect.y() < 0) rect.setY(0);
        if (rect.x() + rect.width() > src.m().cols) rect.setWidth(src.m().cols - rect.x());
        if (rect.y() + rect.width() > src.m().rows) rect.setHeight(src.m().rows - rect.y());

        dst = Mat(src, OpenCVUtils::toRect(rect));
    }
};

BR_REGISTER(Transform, CropFromLandmarksTransform)

} // namespace br

#include "imgproc/cropfromlandmarks.moc"
