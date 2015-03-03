#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Averages a set of landmarks into a new landmark
 * \author Brendan Klare \cite bklare
 */
class AveragePointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    Q_PROPERTY(QString metaName READ get_metaName WRITE set_metaName RESET reset_metaName STORED true)
    Q_PROPERTY(bool append READ get_append WRITE set_append RESET reset_append STORED true)
    Q_PROPERTY(int nLandmarks READ get_nLandmarks WRITE set_nLandmarks RESET reset_nLandmarks STORED true)
    BR_PROPERTY(QList<int>, indices, QList<int>())
    BR_PROPERTY(QString, metaName, "")
    BR_PROPERTY(bool, append, false)
    BR_PROPERTY(int, nLandmarks, 51)

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        if (src.points().size() != nLandmarks) {
            if (Globals->verbose)
                qDebug() << "Warning: Face has " << src.points().size() << "points; should be " << nLandmarks;
            dst.fte = true;
            return;
        }
        int x1 = 0,
            y1 = 0;

        for (int i = 0; i < indices.size(); i++) {
            x1 += src.points()[indices[i]].x();
            y1 += src.points()[indices[i]].y();
        }

        QPointF p(x1 / indices.size(), y1 / indices.size());
        if (!metaName.isEmpty())
            dst.set(metaName, p);
        if (append)
            dst.appendPoint(p);
    }
};

BR_REGISTER(Transform, AveragePointsTransform)

} // namespace br

#include "metadata/averagepoints.moc"
