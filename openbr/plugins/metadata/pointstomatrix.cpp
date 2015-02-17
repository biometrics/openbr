#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Converts either the file::points() list or a QList<QPointF> metadata item to be the template's matrix
 * \author Scott Klum \cite sklum
 */
class PointsToMatrixTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    BR_PROPERTY(QString, inputVariable, QString())

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        if (inputVariable.isEmpty()) {
            dst.m() = OpenCVUtils::pointsToMatrix(dst.file.points());
        } else {
            if (src.file.contains(inputVariable))
                dst.m() = OpenCVUtils::pointsToMatrix(dst.file.get<QList<QPointF> >(inputVariable));
        }
    }
};

BR_REGISTER(Transform, PointsToMatrixTransform)

} // namespace br

#include "metadata/pointstomatrix.moc"
