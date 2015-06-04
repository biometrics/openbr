#include <Eigen/Dense>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/eigenutils.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Designed for use after eye detection + Stasm, this will
 * revert the detected landmarks to the original coordinate space
 * before affine alignment to the stasm mean shape. The storeAffine
 * parameter must be set to true when calling AffineTransform before this.
 * \author Brendan Klare \cite bklare
 */
class RevertAffineTransform : public UntrainableTransform
{
    Q_OBJECT

private:

    void project(const Template &src, Template &dst) const
    {
        QList<float> paramList = src.file.getList<float>("affineParameters");
        Eigen::MatrixXf points = EigenUtils::pointsToMatrix(src.file.points(), true);
        Eigen::MatrixXf affine = Eigen::MatrixXf::Zero(3, 3);
        for (int i = 0, cnt = 0; i < 2; i++)
            for (int j = 0; j < 3; j++, cnt++)
                affine(i, j) = paramList[cnt];
        affine(2, 2) = 1;
        affine = affine.inverse();
        Eigen::MatrixXf affineInv = affine.block(0, 0, 2, 3);
        Eigen::MatrixXf pointsT = points.transpose();
        points =  affineInv * pointsT;
        dst = src;
        dst.file.clearPoints();
        dst.file.setPoints(EigenUtils::matrixToPoints(points.transpose()));
    }
};

BR_REGISTER(Transform, RevertAffineTransform)

} // namespace br

#include "imgproc/revertaffine.moc"
