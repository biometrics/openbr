#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Scales a floating point metadata item by index
 * \br_property int index Index of rect to normalize points with. If negative, will normalize the points to the width of the matrix. Default is -1.
 * \author Scott Klum \cite sklum
 */
class NormalizePointsToRectTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(int index READ get_index WRITE set_index RESET reset_index STORED false)
    BR_PROPERTY(int, index, -1)

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        QList<QPointF> points = src.file.points();
        QList<QPointF> normalizedPoints;

        if (index < 0) {
            qreal width = src.m().cols;
            qreal height = src.m().rows;

            for (int i=0; i<points.size(); i++)
                normalizedPoints.append(QPointF(points[i].x()/width,points[i].y()/height));
        } else {
            QRectF rect = src.file.rects()[index];
            for (int i=0; i<points.size(); i++)
                normalizedPoints.append(QPointF((points[i].x()-rect.left())/rect.width(),
                                                (points[i].y()-rect.right())/rect.height()));
        }

        dst.file.setPoints(normalizedPoints);
    }
};

BR_REGISTER(Transform, NormalizePointsToRectTransform)

} // namespace br

#include "metadata/normalizepointstorect.moc"
