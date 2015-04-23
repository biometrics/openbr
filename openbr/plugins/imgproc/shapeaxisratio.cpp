#include "openbr/plugins/openbr_internal.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/eigenutils.h"

#include <opencv2/contrib/contrib.hpp>

#include <Eigen/Dense>

using namespace Eigen;
using namespace cv;

namespace br
{

class ShapeAxisRatioTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        Mat indices;
        findNonZero(src,indices);

        dst.m() = Mat(1,1,CV_32FC1);

        if (indices.total() > 0) {
            MatrixXd data(indices.total(),2);

            for (size_t i=0; i<indices.total(); i++) {
                data(i,0) = indices.at<Point>(i).y;
                data(i,1) = indices.at<Point>(i).x;
            }

            MatrixXd centered = data.rowwise() - data.colwise().mean();
            MatrixXd cov = (centered.adjoint() * centered) / double(data.rows() - 1);

            SelfAdjointEigenSolver<Eigen::MatrixXd> eSolver(cov);
            MatrixXd D = eSolver.eigenvalues();

            if (eSolver.info() == Success)
                dst.m().at<float>(0,0) = D(0)/D(1);
            else
                dst.file.fte = true;
        } else {
            dst.file.fte = true;
            qWarning("No mask content for %s.",qPrintable(src.file.baseName()));
        }
    }
};

BR_REGISTER(Transform, ShapeAxisRatioTransform)

} // namespace br

#include "imgproc/shapeaxisratio.moc"
