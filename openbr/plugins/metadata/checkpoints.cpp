#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Checks the points in a template for missing (-1,-1) values
 * \author Scott Klum \cite sklum
 * \br_property QList<int> indices Indices of points to check.
 */
class CheckPointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    BR_PROPERTY(QList<int>, indices, QList<int>())

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        for (int i=0; i<indices.size(); i++)
            if (src.points()[indices[i]] == QPointF(-1,-1)) {
                dst.fte = true;
                break;
            }
    }
};

BR_REGISTER(Transform, CheckPointsTransform)

} // namespace br

#include "metadata/checkpoints.moc"
