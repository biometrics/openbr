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
    Q_PROPERTY(bool boundingRect READ get_boundingRect WRITE set_boundingRect RESET reset_boundingRect STORED false)
    BR_PROPERTY(int, index, -1)
    BR_PROPERTY(bool, boundingRect, false)

    QList<QPointF> normalizePoints(const QList<QPointF> &points, const QRectF &rect) const
    {
        QList<QPointF> normalizedPoints;
        for (int i=0; i<points.size(); i++)
            normalizedPoints.append(QPointF((points[i].x()-rect.left())/rect.width(),
                                            (points[i].y()-rect.top())/rect.height()));
        return normalizedPoints;
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        QList<QPointF> points = src.file.points();
        QList<QPointF> normalizedPoints;

        if (boundingRect) {
            float minX, minY;
            minX = minY = std::numeric_limits<int>::max();
            float maxX, maxY;
            maxX = maxY = -std::numeric_limits<int>::max();

            for (int i = 0; i < points.size(); i++) {
                if (points[i].x() < minX) minX = points[i].x();
                if (points[i].x() > maxX) maxX = points[i].x();
                if (points[i].y() < minY) minY = points[i].y();
                if (points[i].y() > maxY) maxY = points[i].y();
            }

            const QRectF boundingRect(QPointF(minX,minY),QPointF(maxX,maxY));
            normalizedPoints = normalizePoints(points,boundingRect);
        } else if (index < 0) {
            qreal width = src.m().cols;
            qreal height = src.m().rows;

            for (int i=0; i<points.size(); i++)
                normalizedPoints.append(QPointF(points[i].x()/width,points[i].y()/height));
        } else
            normalizedPoints = normalizePoints(points, src.file.rects()[index]);

        dst.file.setPoints(normalizedPoints);
    }
};

BR_REGISTER(Transform, NormalizePointsToRectTransform)

} // namespace br

#include "metadata/normalizepointstorect.moc"
