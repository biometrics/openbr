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

    Q_PROPERTY(float normReduction READ get_normReduction WRITE set_normReduction RESET reset_normReduction STORED false)
    Q_PROPERTY(bool center READ get_center WRITE set_center RESET reset_center STORED false)
    Q_PROPERTY(bool warp READ get_warp WRITE set_warp RESET reset_warp STORED false)
    BR_PROPERTY(float, normReduction, 1)
    BR_PROPERTY(bool, center, true)
    BR_PROPERTY(bool, warp, true)

    Eigen::MatrixXf meanShape;

    void train(const TemplateList &data)
    {
        QList< QList<QPointF> > normalizedPoints;

        // Normalize all sets of points
        foreach (br::Template datum, data) {
            QList<QPointF> points = datum.file.points();
            QList<QRectF> rects = datum.file.rects();

            if (points.empty() || rects.empty()) continue;

            // Assume rect appended last was bounding box
            points.append(rects.last().topLeft());
            points.append(rects.last().topRight());
            points.append(rects.last().bottomLeft());
            points.append(rects.last().bottomRight());

            // Center shape at origin
            Scalar mean = cv::mean(OpenCVUtils::toPoints(points).toVector().toStdVector());
            for (int i = 0; i < points.size(); i++) points[i] -= QPointF(mean[0],mean[1]);

            // Remove scale component
            float norm = cv::norm(OpenCVUtils::toPoints(points).toVector().toStdVector());
            for (int i = 0; i < points.size(); i++) {
                points[i] /= (norm/normReduction);
                if (center) points[i] += QPointF(datum.m().cols/2,datum.m().rows/2);
            }

            normalizedPoints.append(points);
        }

        if (normalizedPoints.empty()) qFatal("Unable to calculate normalized points");

        // Determine mean shape, assuming all shapes contain the same number of points
        meanShape = Eigen::MatrixXf(normalizedPoints[0].size(), 2);

        for (int i = 0; i < normalizedPoints[0].size(); i++) {
            double x = 0;
            double y = 0;

            for (int j = 0; j < normalizedPoints.size(); j++) {
                x += normalizedPoints[j][i].x();
                y += normalizedPoints[j][i].y();
            }

            x /= (double)normalizedPoints.size();
            y /= (double)normalizedPoints.size();

            meanShape(i,0) = x;
            meanShape(i,1) = y;
        }
    }

    void project(const Template &src, Template &dst) const
    {
        QList<QPointF> points = src.file.points();
        QList<QRectF> rects = src.file.rects();

        if (points.empty() || rects.empty()) {
            dst = src;
            qWarning("Procrustes alignment failed because points or rects are empty.");
            return;
        }

        // Assume rect appended last was bounding box
        points.append(rects.last().topLeft());
        points.append(rects.last().topRight());
        points.append(rects.last().bottomLeft());
        points.append(rects.last().bottomRight());

        Scalar mean = cv::mean(OpenCVUtils::toPoints(points).toVector().toStdVector());
        for (int i = 0; i < points.size(); i++) points[i] -= QPointF(mean[0],mean[1]);

        Eigen::MatrixXf srcMat(points.size(), 2);
        float norm = cv::norm(OpenCVUtils::toPoints(points).toVector().toStdVector());
        for (int i = 0; i < points.size(); i++) {
            points[i] /= (norm/normReduction);
            if (center) points[i] += QPointF(src.m().cols/2,src.m().rows/2);
            srcMat(i,0) = points[i].x();
            srcMat(i,1) = points[i].y();
        }

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(srcMat.transpose()*meanShape, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXf R = svd.matrixU()*svd.matrixV().transpose();

        dst = src;

        if (warp) {
            Eigen::MatrixXf dstMat = srcMat*R;
            for (int i = 0; i < dstMat.rows(); i++) {
                dst.file.appendPoint(QPointF(dstMat(i,0),dstMat(i,1)));
            }
        }

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
 * \brief Creates a Delaunay triangulation based on a set of points
 * \author Scott Klum \cite sklum
 */
class DelaunayTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(float normReduction READ get_normReduction WRITE set_normReduction RESET reset_normReduction STORED false)
    Q_PROPERTY(bool warp READ get_warp WRITE set_warp RESET reset_warp STORED false)
    Q_PROPERTY(bool draw READ get_draw WRITE set_draw RESET reset_draw STORED false)
    BR_PROPERTY(float, normReduction, 1)
    BR_PROPERTY(bool, warp, true)
    BR_PROPERTY(bool, draw, false)

    void project(const Template &src, Template &dst) const
    {
        Subdiv2D subdiv(Rect(0,0,src.m().cols,src.m().rows));

        QList<QPointF> points = src.file.points();
        QList<QRectF> rects = src.file.rects();

        if (points.empty() || rects.empty()) {
            dst = src;
            qWarning("Delauney triangulation failed because points or rects are empty.");
            return;
        }

        // Assume rect appended last was bounding box
        points.append(rects.last().topLeft());
        points.append(rects.last().topRight());
        points.append(rects.last().bottomLeft());
        points.append(rects.last().bottomRight());

        for (int i = 0; i < points.size(); i++) {
            if (points[i].x() < 0 || points[i].y() < 0 || points[i].y() >= src.m().rows || points[i].x() >= src.m().cols) {
                dst = src;
                qWarning("Delauney triangulation failed because points lie on boundary.");
                return;
            }
            subdiv.insert(OpenCVUtils::toPoint(points[i]));
        }

        vector<Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);

        QList< QList<Point> > validTriangles;

        for (size_t i = 0; i < triangleList.size(); ++i) {

            vector<Point> pt(3);
            pt[0] = Point(cvRound(triangleList[i][0]), cvRound(triangleList[i][1]));
            pt[1] = Point(cvRound(triangleList[i][2]), cvRound(triangleList[i][3]));
            pt[2] = Point(cvRound(triangleList[i][4]), cvRound(triangleList[i][5]));

            bool inside = true;
            for (int j = 0; j < 3; j++) {
                if (pt[j].x > src.m().cols || pt[j].y > src.m().rows || pt[j].x <= 0 || pt[j].y <= 0) inside = false;
            }

            if (inside) validTriangles.append(QList<Point>()<< pt[0] << pt[1] << pt[2]);
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
                    srcMat(j,0) = (validTriangles[i][j].x-mean[0])/(norm/normReduction)+src.m().cols/2;
                    srcMat(j,1) = (validTriangles[i][j].y-mean[1])/(norm/normReduction)+src.m().rows/2;
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

                dst.m() += output;
            }

            Rect boundingBox = boundingRect(mappedPoints.toVector().toStdVector());

            boundingBox.x += 0; //boundingBox.width * .05;
            boundingBox.y += boundingBox.height * .1; // 0.025 for nose, .05 for mouth, .10 for brow
            boundingBox.width *= 1;//.975;
            boundingBox.height *= .80; // .975 for nose, .95 for mouth, .925 for brow

            qDebug() << boundingBox;

            dst.m() = Mat(dst.m(), boundingBox);
        }
    }

};

BR_REGISTER(Transform, DelaunayTransform)

/*!
 * \ingroup transforms
 * \brief Computes the mean of a set of templates.
 * \note Suitable for visualization only as it sets every projected template to the mean template.
 * \author Scott Klum \cite sklum
 */
class MeanTransform : public Transform
{
    Q_OBJECT

    Mat mean;

    void train(const TemplateList &data)
    {
        mean = Mat::zeros(data[0].m().rows,data[0].m().cols,CV_32F);

        for (int i = 0; i < data.size(); i++) {
            Mat converted;
            data[i].m().convertTo(converted, CV_32F);
            mean += converted;
        }

        mean /= data.size();
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
