#include <opencv2/opencv.hpp>
#include "openbr_internal.h"
#include "openbr/core/qtutils.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/eigenutils.h"
#include <QString>
#include <Eigen/SVD>

using namespace std;
using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Procrustes alignment of points
 * \author Scott Klum \cite sklum
 */
class ProcrustesTransform : public Transform
{
    Q_OBJECT

    Eigen::MatrixXf meanShape;

    void train(const TemplateList &data)
    {
        QList< QList<QPointF> > normalizedPoints;

        // Normalize all sets of points
        foreach (br::Template datum, data) {
            QList<QPointF> points = datum.file.points();

            if (points.empty()) continue;

            QList<QRectF> rects = datum.file.rects();

            if (rects.size() > 1) qWarning("More than one rect in training data templates.");

            points.append(rects[0].topLeft());
            points.append(rects[0].topRight());
            points.append(rects[0].bottomLeft());
            points.append(rects[0].bottomRight());

            cv::Scalar mean = cv::mean(OpenCVUtils::toPoints(points).toVector().toStdVector());
            for (int i = 0; i < points.size(); i++) points[i] -= QPointF(mean[0],mean[1]);

            float norm = cv::norm(OpenCVUtils::toPoints(points).toVector().toStdVector());
            for (int i = 0; i < points.size(); i++) points[i] /= norm;

            normalizedPoints.append(points);
        }

        // Determine mean shape
        Eigen::MatrixXf shapeBuffer(normalizedPoints[0].size(), 2);

        for (int i = 0; i < normalizedPoints[0].size(); i++) {

            double x = 0;
            double y = 0;

            for (int j = 0; j < normalizedPoints.size(); j++) {
                x += normalizedPoints[j][i].x();
                y += normalizedPoints[j][i].y();
            }

            x /= (double)normalizedPoints.size();
            y /= (double)normalizedPoints.size();

            shapeBuffer(i,0) = x;
            shapeBuffer(i,1) = y;
        }

        meanShape = shapeBuffer;
    }

    void project(const Template &src, Template &dst) const
    {
        dst.m() = src.m();

        QList<QPointF> points = src.file.points();
        QList<QRectF> rects = src.file.rects();

        if (points.empty() || rects.empty()) {
            dst = src;
            qWarning("Points/rects are empty.");
            return;
        }

        if (rects.size() > 1) qWarning("More than one rect in training data templates.");

        points.append(rects[0].topLeft());
        points.append(rects[0].topRight());
        points.append(rects[0].bottomLeft());
        points.append(rects[0].bottomRight());

        cv::Scalar mean = cv::mean(OpenCVUtils::toPoints(points).toVector().toStdVector());
        for (int i = 0; i < points.size(); i++) points[i] -= QPointF(mean[0],mean[1]);

        float norm = cv::norm(OpenCVUtils::toPoints(points).toVector().toStdVector());
        Eigen::MatrixXf srcPoints(points.size(), 2);

        for (int i = 0; i < points.size(); i++) {
            srcPoints(i,0) = points[i].x()/(norm/100.)+50;
            srcPoints(i,1) = points[i].y()/(norm/100.)+50;
        }

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(srcPoints.transpose()*meanShape, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::MatrixXf R = svd.matrixU()*svd.matrixV().transpose();

        dst.file.set("Procrustes_0_0", R(0,0));
        dst.file.set("Procrustes_1_0", R(1,0));
        dst.file.set("Procrustes_1_1", R(1,1));
        dst.file.set("Procrustes_0_1", R(0,1));
        dst.file.set("Procrustes_mean_0", mean[0]);
        dst.file.set("Procrustes_mean_1", mean[1]);
        dst.file.set("Procrustes_norm", norm);
    }

    void store(QDataStream &stream) const
    {
        stream << meanShape;
    }

    void load(QDataStream &stream)
    {
        stream >> meanShape;
    }

};

BR_REGISTER(Transform, ProcrustesTransform)

/*!
 * \ingroup transforms
 * \brief Creates a delauney triangulation based on a set of points
 * \author Scott Klum \cite sklum
 */
class DelaunayTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(bool draw READ get_draw WRITE set_draw RESET reset_draw STORED false)
    BR_PROPERTY(bool, draw, false)

    void project(const Template &src, Template &dst) const
    {

        Subdiv2D subdiv(Rect(0,0,src.m().cols,src.m().rows));

        QList<cv::Point2f> points = OpenCVUtils::toPoints(src.file.points());
        QList<QRectF> rects = src.file.rects();

        if (points.empty() || rects.empty()) {
            dst = src;
            qWarning("Points/rects are empty.");
            return;
        }

        if (rects.size() > 1) qWarning("More than one rect in training data templates.");

        points.append(OpenCVUtils::toPoint(rects[0].topLeft()));
        points.append(OpenCVUtils::toPoint(rects[0].topRight()));
        points.append(OpenCVUtils::toPoint(rects[0].bottomLeft()));
        points.append(OpenCVUtils::toPoint(rects[0].bottomRight()));

        for (int i = 0; i < points.size(); i++) subdiv.insert(points[i]);

        vector<Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);

        QList< QList<Point> > validTriangles;
        int count = 0;

        for (size_t i = 0; i < triangleList.size(); ++i) {
            vector<Point> pt(3);

            Vec6f t = triangleList[i];

            pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
            pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
            pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

            bool inside = true;
            for (int j = 0; j < 3; j++) {
                if (pt[j].x > src.m().cols || pt[j].y > src.m().rows || pt[j].x <= 0 || pt[j].y <= 0) inside = false;
            }

            if (inside) {
                count++;

                QList<Point> triangleBuffer;

                triangleBuffer << pt[0] << pt[1] << pt[2];

                validTriangles.append(triangleBuffer);
            }
        }

        dst.m() = src.m().clone();

        if (draw) {
            foreach(const QList<Point>& triangle, validTriangles) {
                line(dst.m(), triangle[0], triangle[1], Scalar(0,0,0), 1);
                line(dst.m(), triangle[1], triangle[2], Scalar(0,0,0), 1);
                line(dst.m(), triangle[2], triangle[0], Scalar(0,0,0), 1);
            }
        }

        bool warp = true;

        if (warp) {
            Eigen::MatrixXf R(2,2);
            R(0,0) = src.file.get<float>("Procrustes_0_0");
            R(1,0) = src.file.get<float>("Procrustes_1_0");
            R(1,1) = src.file.get<float>("Procrustes_1_1");
            R(0,1) = src.file.get<float>("Procrustes_0_1");

            cv::Scalar mean(2);
            mean[0] = src.file.get<float>("Procrustes_mean_0");
            mean[1] = src.file.get<float>("Procrustes_mean_1");

            float norm = src.file.get<float>("Procrustes_norm");

            dst.m() = Mat::zeros(src.m().rows,src.m().cols,src.m().type());

            QList<Point2f> mappedPoints;

            for (int i = 0; i < validTriangles.size(); i++) {
                Eigen::MatrixXf srcMat(validTriangles[i].size(), 2);

                for (int j = 0; j < validTriangles[i].size(); j++) {
                    srcMat(j,0) = (validTriangles[i][j].x-mean[0])/(norm/100.)+50;
                    srcMat(j,1) = (validTriangles[i][j].y-mean[1])/(norm/100.)+50;
                }

                Eigen::MatrixXf dstMat = srcMat*R;

                Point2f srcPoints[3];
                for (int j = 0; j < 3; j++) srcPoints[j] = validTriangles[i][j];

                Point2f dstPoints[3];
                for (int j = 0; j < 3; j++) {
                    dstPoints[j] = Point2f(dstMat(j,0),dstMat(j,1));
                    mappedPoints.append(dstPoints[j]);
                }

                Mat buffer(src.m().rows,src.m().cols,src.m().type());

                warpAffine(src.m(), buffer, getAffineTransform(srcPoints, dstPoints), Size(src.m().cols,src.m().rows));

                Mat mask = Mat::zeros(src.m().rows, src.m().cols, CV_8UC1);
                Point maskPoints[1][3];
                maskPoints[0][0] = dstPoints[0];
                maskPoints[0][1] = dstPoints[1];
                maskPoints[0][2] = dstPoints[2];
                const Point* ppt = { maskPoints[0] };

                fillConvexPoly(mask, ppt, 3, Scalar(255,255,255), 8);

                Mat output(src.m().rows,src.m().cols,src.m().type());

                // Optimize
                if (i > 0) {
                    Mat overlap;
                    bitwise_and(dst.m(),mask,overlap);
                    for (int j = 0; j < overlap.rows; j++) {
                        for (int k = 0; k < overlap.cols; k++) {
                            if (overlap.at<uchar>(k,j) != 0) {
                                mask.at<uchar>(k,j) = 0;
                            }
                        }
                    }
                }

                bitwise_and(buffer,mask,output);

                dst.m() += output; // Need to xor stuff out
            }

            Rect boundingBox = boundingRect(mappedPoints.toVector().toStdVector());
            boundingBox.x += boundingBox.width * .025;
            boundingBox.y += boundingBox.height * .025; // 0.025 for nose, .05 for mouth, .10 for brow
            boundingBox.width *= .975;
            boundingBox.height *= .975; // .975 for nose, .95 for mouth, .925 for brow

            dst.m() = Mat(dst.m(), boundingBox);
        }
    }

};

BR_REGISTER(Transform, DelaunayTransform)

/*!
 * \ingroup transforms
 * \brief Wraps STASM key point detector
 * \author Scott Klum \cite sklum
 */
class MeanTransform : public Transform
{
    Q_OBJECT

    Mat mean;

    void train(const TemplateList &data)
    {
        mean = Mat::zeros(data[0].m().rows,data[0].m().cols,CV_32F);

        for (int i = 0; i < data.size()/2; i++) {
            Mat converted;
            data[i].m().convertTo(converted, CV_32F);
            mean += converted;
        }

        mean /= data.size()/2;
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.m() = mean;
    }

};

BR_REGISTER(Transform, MeanTransform)

} // namespace br

#include "landmarks.moc"
