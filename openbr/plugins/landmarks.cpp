#include <opencv2/opencv.hpp>
#include "openbr_internal.h"
#include "openbr/core/qtutils.h"
#include "openbr/core/opencvutils.h"
#include <QString>
#include <Eigen/SVD>

using namespace std;
using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps STASM key point detector
 * \author Scott Klum \cite sklum
 */
class ProcrustesTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(QString principalShapePath READ get_principalShapePath WRITE set_principalShapePath RESET reset_principalShapePath STORED false)
    BR_PROPERTY(QString, principalShapePath, QString())

    Eigen::MatrixXf meanShape;
    Mat shapeMat;

    void train(const TemplateList &data)
    {
        QList< QList<cv::Point2f> > normalizedPoints;

        // Normalize all sets of points
        foreach (br::Template datum, data) {
            QList<cv::Point2f> points = OpenCVUtils::toPoints(datum.file.points());

            if (points.empty()) {
                continue;
            }

            cv::Scalar mean = cv::mean(points.toVector().toStdVector());
            for (int i = 0; i < points.size(); i++) {
                points[i].x -= mean[0];
                points[i].y -= mean[1];
            }

            float norm = cv::norm(points.toVector().toStdVector());
            for (int i = 0; i < points.size(); i++) {
                points[i].x /= (norm);
                points[i].y /= (norm);
            }

            normalizedPoints.append(points);
        }

        // Determine mean shape
        Eigen::MatrixXf shapeTest(normalizedPoints[0].size(), 2);

        cv::Mat shapeBuffer(normalizedPoints[0].size(), 2, CV_32F);

        for (int i = 0; i < normalizedPoints[0].size(); i++) {

            double x = 0;
            double y = 0;

            for (int j = 0; j < normalizedPoints.size(); j++) {
                x += normalizedPoints[j][i].x;
                y += normalizedPoints[j][i].y;
            }

            x /= (double)normalizedPoints.size();
            y /= (double)normalizedPoints.size();

            shapeBuffer.at<float>(i,0) = x;
            shapeBuffer.at<float>(i,1) = y;

            shapeTest(i,0) = x;
            shapeTest(i,1) = y;
        }

        meanShape = shapeTest;
    }

    void project(const Template &src, Template &dst) const
    {
        QList<QPointF> points = src.file.points();

        cv::Scalar mean = cv::mean(OpenCVUtils::toPoints(points).toVector().toStdVector());

        for (int i = 0; i < points.size(); i++) points[i] -= QPointF(mean[0],mean[1]);

        float norm = cv::norm(OpenCVUtils::toPoints(points).toVector().toStdVector());

        Eigen::MatrixXf srcPoints(points.size(), 2);

        for (int i = 0; i < points.size(); i++) {
            srcPoints(i,0) = points[i].x()/(norm/150.)+50;
            srcPoints(i,1) = points[i].y()/(norm/150.)+50;
        }

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(srcPoints.transpose()*meanShape, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::MatrixXf R = svd.matrixU()*svd.matrixV().transpose();

        Eigen::MatrixXf dstPoints = srcPoints*R;

        points.clear();

        for (int i = 0; i < dstPoints.rows(); i++) points.append(QPointF(dstPoints(i,0),dstPoints(i,1)));

        dst.file.setPoints(points);
    }

};

BR_REGISTER(Transform, ProcrustesTransform)

/*!
 * \ingroup transforms
 * \brief Wraps STASM key point detector
 * \author Scott Klum \cite sklum
 */
class DelauneyTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(bool draw READ get_draw WRITE set_draw RESET reset_draw STORED false)
    BR_PROPERTY(bool, draw, false)

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        Subdiv2D subdiv(Rect(0,0,src.m().cols,src.m().rows));

                                     foreach(const cv::Point2f& point, OpenCVUtils::toPoints(src.file.points())) subdiv.insert(point);

        vector<Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);
        vector<Point> pt(3);

        Scalar delaunay_color(0, 0, 0);

        if (draw) {
            for(size_t i = 0; i < triangleList.size(); ++i) {
                Vec6f t = triangleList[i];

                pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
                pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
                pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
                bool outside = true;
                for (int i = 0; i < 3; i++) {
                    if(pt[i].x > dst.m().cols || pt[i].y > dst.m().rows || pt[i].x <= 0 || pt[i].y <= 0)
                        outside = false;
                }
                if (outside) {
                    line(dst.m(), pt[0], pt[1], delaunay_color, 1);
                    line(dst.m(), pt[1], pt[2], delaunay_color, 1);
                    line(dst.m(), pt[2], pt[0], delaunay_color, 1);
                }
            }
        }
    }

};

BR_REGISTER(Transform, DelauneyTransform)

} // namespace br

#include "landmarks.moc"
