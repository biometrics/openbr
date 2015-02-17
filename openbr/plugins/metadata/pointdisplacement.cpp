#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Normalize points to be relative to a single point
 * \author Scott Klum \cite sklum
 */
class PointDisplacementTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        QList<QPointF> points = dst.points();
        QList<QPointF> normalizedPoints;

        for (int i=0; i<points.size(); i++)
            for (int j=0; j<points.size(); j++)
                // There is redundant information here
                if (j!=i) {
                    QPointF normalizedPoint = points[i]-points[j];
                    normalizedPoint.setX(pow(normalizedPoint.x(),2));
                    normalizedPoint.setY(pow(normalizedPoint.y(),2));
                    normalizedPoints.append(normalizedPoint);
                }

        dst.setPoints(normalizedPoints);
    }
};

BR_REGISTER(Transform, PointDisplacementTransform)

} // namespace br

#include "metadata/pointdisplacement.moc"
