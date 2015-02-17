#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Name a point/rect
 * \author Scott Klum \cite sklum
 */
class NameLandmarksTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(bool point READ get_point WRITE set_point RESET reset_point STORED false)
    BR_PROPERTY(bool, point, true)
    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    Q_PROPERTY(QStringList names READ get_names WRITE set_names RESET reset_names STORED false)
    BR_PROPERTY(QList<int>, indices, QList<int>())
    BR_PROPERTY(QStringList, names, QStringList())

    void projectMetadata(const File &src, File &dst) const
    {
        if (indices.size() != names.size()) qFatal("Index/name size mismatch");

        dst = src;

        if (point) {
            QList<QPointF> points = src.points();

            for (int i=0; i<indices.size(); i++) {
                if (indices[i] < points.size()) dst.set(names[i], points[indices[i]]);
                else qFatal("Index out of range.");
            }
        } else {
            QList<QRectF> rects = src.rects();

            for (int i=0; i<indices.size(); i++) {
                if (indices[i] < rects.size()) dst.set(names[i], rects[indices[i]]);
                else qFatal("Index out of range.");
            }
        }
    }
};

BR_REGISTER(Transform, NameLandmarksTransform)

} // namespace br

#include "metadata/namelandmarks.moc"
