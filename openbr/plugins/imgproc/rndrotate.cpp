#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Randomly rotates an image in a specified range.
 * \author Scott Klum \cite sklum
 */
class RndRotateTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QList<int> range READ get_range WRITE set_range RESET reset_range STORED false)
    BR_PROPERTY(QList<int>, range, QList<int>() << -15 << 15)

    void project(const Template &src, Template &dst) const {
        int span = range.first() - range.last();
        int angle = (rand() % span) + range.first();
        Mat rotMatrix = getRotationMatrix2D(Point2f(src.m().rows/2,src.m().cols/2),angle,1.0);
        warpAffine(src,dst,rotMatrix,Size(src.m().cols,src.m().rows));

        QList<QPointF> points = src.file.points();
        QList<QPointF> rotatedPoints;
        for (int i=0; i<points.size(); i++) {
            rotatedPoints.append(QPointF(points.at(i).x()*rotMatrix.at<double>(0,0)+
                                         points.at(i).y()*rotMatrix.at<double>(0,1)+
                                         rotMatrix.at<double>(0,2),
                                         points.at(i).x()*rotMatrix.at<double>(1,0)+
                                         points.at(i).y()*rotMatrix.at<double>(1,1)+
                                         rotMatrix.at<double>(1,2)));
        }

        dst.file.setPoints(rotatedPoints);
    }
};

BR_REGISTER(Transform, RndRotateTransform)

} // namespace br

#include "imgproc/rndrotate.moc"
