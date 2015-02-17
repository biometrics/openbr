#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Normalize points to be relative to a single point
 * \author Scott Klum \cite sklum
 */
class NormalizePointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    Q_PROPERTY(int index READ get_index WRITE set_index RESET reset_index STORED false)
    BR_PROPERTY(int, index, 0)

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        QList<QPointF> points = dst.points();
        QPointF normPoint = points.at(index);

        QList<QPointF> normalizedPoints;

        for (int i=0; i<points.size(); i++)
            if (i!=index)
                normalizedPoints.append(normPoint-points[i]);

        dst.setPoints(normalizedPoints);
    }
};

BR_REGISTER(Transform, NormalizePointsTransform)

} // namespace br

#include "metadata/normalizepoints.moc"
