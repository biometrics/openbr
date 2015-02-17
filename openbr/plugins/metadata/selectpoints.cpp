#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Retains only landmarks/points at the provided indices
 * \author Brendan Klare \cite bklare
 */
class SelectPointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    Q_PROPERTY(bool invert READ get_invert WRITE set_invert RESET reset_invert STORED false) // keep the points _not_ in the list
    BR_PROPERTY(QList<int>, indices, QList<int>())
    BR_PROPERTY(bool, invert, false)

    void projectMetadata(const File &src, File &dst) const
    {
        const QList<QPointF> srcPoints = src.points();
        QList<QPointF> dstPoints;
        for (int i=0; i<srcPoints.size(); i++)
            if (indices.contains(i) ^ invert)
                dstPoints.append(srcPoints[i]);
        dst.setPoints(dstPoints);
    }
};

BR_REGISTER(Transform, SelectPointsTransform)

} // namespace br

#include "metadata/selectpoints.moc"
