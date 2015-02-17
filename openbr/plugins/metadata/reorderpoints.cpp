#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Reorder the points such that points[from[i]] becomes points[to[i]] and
 *        vice versa
 * \author Scott Klum \cite sklum
 */
class ReorderPointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    Q_PROPERTY(QList<int> from READ get_from WRITE set_from RESET reset_from STORED false)
    Q_PROPERTY(QList<int> to READ get_to WRITE set_to RESET reset_to STORED false)
    BR_PROPERTY(QList<int>, from, QList<int>())
    BR_PROPERTY(QList<int>, to, QList<int>())

    void projectMetadata(const File &src, File &dst) const
    {
        if (from.size() == to.size()) {
            QList<QPointF> points = src.points();
            int size = src.points().size();
            if (!points.contains(QPointF(-1,-1)) && Common::Max(from) < size && Common::Max(to) < size) {
                for (int i=0; i<from.size(); i++) {
                    std::swap(points[from[i]],points[to[i]]);
                }
                dst.setPoints(points);
            }
        } else qFatal("Inconsistent sizes for to and from index lists.");
    }
};

BR_REGISTER(Transform, ReorderPointsTransform)

} // namespace br

#include "metadata/reorderpoints.moc"
