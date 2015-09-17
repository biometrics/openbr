#include <openbr/plugins/openbr_internal.h>
#include <cmath>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Calculate interpupillary distance or the distance between two arbitrary points.
 * \author Ben Klein \cite bhklein
 * \br_property bool named Are the points named?
 * \br_property QString firstEye First point's metadata key.
 * \br_property QString secondEye Second point's metadata key.
 * \br_property QString key Metadata key for distance.
 * \br_property QList<int> indices Indices of points in metadata if not named.
 */
class IPDTransform : public UntrainableMetadataTransform
{
    Q_OBJECT


    Q_PROPERTY(bool named READ get_named WRITE set_named RESET reset_named STORED false)
    Q_PROPERTY(QString firstEye READ get_firstEye WRITE set_firstEye RESET reset_firstEye STORED false)
    Q_PROPERTY(QString secondEye READ get_secondEye WRITE set_secondEye RESET reset_secondEye STORED false)
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    BR_PROPERTY(bool, named, true)
    BR_PROPERTY(QString, firstEye, "First_Eye")
    BR_PROPERTY(QString, secondEye, "Second_Eye")
    BR_PROPERTY(QString, key, "IPD")
    BR_PROPERTY(QList<int>, indices, QList<int>())

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        if (!indices.empty()) {
            if (indices.size() != 2) {
                qDebug() << "Indices must be of length 2 to calculate IPD!";
                return;
            }

            QPointF first = src.points()[indices[0]];
            QPointF second = src.points()[indices[1]];
            float distX = second.x() - first.x();
            float distY = second.y() - first.y();
            float distance = std::sqrt(distX*distX + distY*distY);
            dst.set(key, distance);
        } else {
            QPointF first = src.get<QPointF>(firstEye, QPointF());
            QPointF second = src.get<QPointF>(secondEye, QPointF());
            float distX = second.x() - first.x();
            float distY = second.y() - first.y();
            float distance = std::sqrt(distX*distX + distY*distY);
            dst.set(key, distance);
        }
    }
};

BR_REGISTER(Transform, IPDTransform)

} // namespace br

#include "metadata/ipd.moc"
