#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Creates a Delaunay triangulation based on a set of points
 * \author Scott Klum \cite sklum
 */
class DrawDelaunayTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        if (src.file.contains("DelaunayTriangles")) {
            QList<Point2f> validTriangles = OpenCVUtils::toPoints(src.file.getList<QPointF>("DelaunayTriangles"));

            // Clone the matrix do draw on it
            for (int i = 0; i < validTriangles.size(); i+=3) {
                line(dst, validTriangles[i], validTriangles[i+1], Scalar(0,0,0), 1);
                line(dst, validTriangles[i+1], validTriangles[i+2], Scalar(0,0,0), 1);
                line(dst, validTriangles[i+2], validTriangles[i], Scalar(0,0,0), 1);
            }
        } else qWarning("Template does not contain Delaunay triangulation.");
    }
};

BR_REGISTER(Transform, DrawDelaunayTransform)

} // namespace br

#include "gui/drawdelaunay.moc"
