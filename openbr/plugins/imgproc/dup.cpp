#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Duplicates the template data.
 * \author Josh Klontz \cite jklontz
 */
class DupTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    Q_PROPERTY(bool dupLandmarks READ get_dupLandmarks WRITE set_dupLandmarks RESET reset_dupLandmarks STORED false)
    BR_PROPERTY(int, n, 1)
    BR_PROPERTY(bool, dupLandmarks, false)

    void project(const Template &src, Template &dst) const
    {
        for (int i=0; i<n; i++)
            dst.merge(src);

        if (dupLandmarks) {
            QList<QPointF> points = src.file.points();
            QList<QRectF> rects = src.file.rects();

            for (int i=1; i<n; i++) {
                dst.file.appendPoints(points);
                dst.file.appendRects(rects);
            }
        }
    }
};

BR_REGISTER(Transform, DupTransform)

} // namespace br

#include "imgproc/dup.moc"
