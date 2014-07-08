#include <opencv2/opencv.hpp>
#include "openbr_internal.h"
#include "openbr/core/qtutils.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/eigenutils.h"
#include <QString>
#include <Eigen/SVD>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

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

    Q_PROPERTY(bool warp READ get_warp WRITE set_warp RESET reset_warp STORED false)
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
            for (int i = 0; i < points.size(); i++) points[i] /= norm;

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
            if (Globals->verbose) qWarning("Procrustes alignment failed because points or rects are empty.");
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
            points[i] /= norm;
            srcMat(i,0) = points[i].x();
            srcMat(i,1) = points[i].y();
        }

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(srcMat.transpose()*meanShape, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXf R = svd.matrixU()*svd.matrixV().transpose();

        dst = src;

        // Store procrustes stats in the order:
        // R(0,0), R(1,0), R(1,1), R(0,1), mean_x, mean_y, norm
        QList<float> procrustesStats;
        procrustesStats << R(0,0) << R(1,0) << R(1,1) << R(0,1) << mean[0] << mean[1] << norm;
        dst.file.setList<float>("ProcrustesStats",procrustesStats);

        if (warp) {
            Eigen::MatrixXf dstMat = srcMat*R;
            for (int i = 0; i < dstMat.rows(); i++) {
                dst.file.appendPoint(QPointF(dstMat(i,0),dstMat(i,1)));
            }
        }
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
 * \brief Improved procrustes alignment of points, to include a post processing scaling of points
 * to faciliate subsequent texture mapping.
 * \author Brendan Klare \cite bklare
 */
class ProcrustesAlignTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(float width READ get_width WRITE set_width RESET reset_width STORED false)
    Q_PROPERTY(float padding READ get_padding WRITE set_padding RESET reset_padding STORED false)
    BR_PROPERTY(float, width, 80)
    BR_PROPERTY(float, padding, 8)

    Eigen::MatrixXf referenceShape;
    float minX;
    float minY;
    float maxX;
    float maxY;
    float aspectRatio;

    void init() {
        minX = FLT_MAX,
        minY = FLT_MAX,
        maxX = -FLT_MAX,
        maxY = -FLT_MAX;
        aspectRatio = 0;
    }

    static MatrixXf getRotation(MatrixXf ref, MatrixXf sample) {
        MatrixXf R = ref.transpose() * sample;
        JacobiSVD<MatrixXf> svd(R, ComputeFullU | ComputeFullV);
        R = svd.matrixU() * svd.matrixV();
        return R;
    }

    //Converts x y points in a single vector to two column matrix
    static MatrixXf vectorToMatrix(MatrixXf vector) {
        int n = vector.rows();
        MatrixXf matrix(n / 2, 2);
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < 2; j++) {
                matrix(i, j) = vector(i * 2 + j);
            }
        }
        return matrix;
    }

    static MatrixXf matrixToVector(MatrixXf matrix) {
        int n2 = matrix.rows();
        MatrixXf vector(n2 * 2, 1);
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < 2; j++) {
                vector(i * 2 + j) = matrix(i, j);
            }
        }
        return vector;
    }

    void train(const TemplateList &data)
    {
        MatrixXf points(data[0].file.points().size() * 2, data.size());

        // Normalize all sets of points
        for (int j = 0; j < data.size(); j++) {
            QList<QPointF> imagePoints = data[j].file.points();

            float meanX = 0,
                  meanY = 0;
            for (int i = 0; i < imagePoints.size(); i++) {
                points(i * 2, j) = imagePoints[i].x();
                points(i * 2 + 1, j) = imagePoints[i].y();
                meanX += imagePoints[i].x();
                meanY += imagePoints[i].y();
            }
            meanX /= imagePoints.size();
            meanY /= imagePoints.size();

            for (int i = 0; i < imagePoints.size(); i++) {
                points(i * 2, j) -= meanX;
                points(i * 2 + 1, j) -= meanY;
            }
        }

        //normalize scale
        for (int i = 0; i < points.cols(); i++)
            points.col(i) = points.col(i) / points.col(i).norm();

        //Normalize rotation
        MatrixXf refPrev;
        referenceShape = vectorToMatrix(points.rowwise().sum() / points.cols());
        float diff = FLT_MAX;
        while (diff > 1e-5) {//iterate until reference shape is stable
            refPrev = referenceShape;

            for (int j = 0; j < points.cols(); j++) {
                MatrixXf p = vectorToMatrix(points.col(j));
                MatrixXf R = getRotation(referenceShape, p);
                p = p * R.transpose();
                points.col(j) = matrixToVector(p);
            }
            referenceShape = vectorToMatrix(points.rowwise().sum() / points.cols());
            diff = (matrixToVector(referenceShape) - matrixToVector(refPrev)).norm();
        }

        referenceShape = vectorToMatrix(points.rowwise().sum() / points.cols());

        //Choose crop boundaries and adjustments that captures all data
        for (int i = 0; i < points.rows(); i++) {
            for (int j = 0; j < points.cols(); j++) {
                if (i % 2 == 0) {
                if (points(i,j) > maxX)
                    maxX = points(i, j);
                if (points(i,j) < minX)
                    minX = points(i, j);
                } else {
                if (points(i,j) > maxY)
                    maxY = points(i, j);
                if (points(i,j) < minY)
                    minY = points(i, j);
                }
            }
        }
        aspectRatio = (maxX - minX) / (maxY - minY);
    }

    void project(const Template &src, Template &dst) const
    {
        QList<QPointF> imagePoints = src.file.points();
        MatrixXf p(imagePoints.size() * 2, 1);
        for (int i = 0; i < imagePoints.size(); i++) {
            p(i * 2) = imagePoints[i].x();
            p(i * 2 + 1) = imagePoints[i].y();
        }
        p = vectorToMatrix(p);

        //Normalize translation
        p.col(0) = p.col(0) - MatrixXf::Ones(p.rows(),1) * (p.col(0).sum() / p.rows());
        p.col(1) = p.col(1) - MatrixXf::Ones(p.rows(),1) * (p.col(1).sum() / p.rows());

        //Normalize scale
        p /= matrixToVector(p).norm();

        //Normalize rotation
        MatrixXf R = getRotation(referenceShape, p);
        p = p * R.transpose();

        //Translate and scale into output space and store in output list
        QList<QPointF> procrustesPoints;
        for (int i = 0; i < p.rows(); i++)
            procrustesPoints.append( QPointF(
                (p(i, 0) - minX) / (maxX - minX) * (width - 1) + padding,
                (p(i, 1) - minY) / (maxY - minY) * (qRound( width / aspectRatio) - 1) + padding));

        dst = src;
        dst.file.setList<QPointF>("ProcrustesPoints", procrustesPoints);
        dst.file.set("ProcrustesBound", QRectF(0, 0, width + 2 * padding, (qRound(width / aspectRatio) + 2 * padding)));
    }

    void store(QDataStream &stream) const
    {
        stream << referenceShape;
        stream << minX;
        stream << minY;
        stream << maxX;
        stream << maxY;
        stream << aspectRatio;
    }

    void load(QDataStream &stream)
    {
        stream >> referenceShape;
        stream >> minX;
        stream >> minY;
        stream >> maxX;
        stream >> maxY;
        stream >> aspectRatio;
    }
};

BR_REGISTER(Transform, ProcrustesAlignTransform)

/*!
 * \ingroup transforms
 * \brief Creates a Delaunay triangulation based on a set of points
 * \author Scott Klum \cite sklum
 */
class DelaunayTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(bool warp READ get_warp WRITE set_warp RESET reset_warp STORED false)
    BR_PROPERTY(float, scaleFactor, 1)
    BR_PROPERTY(bool, warp, true)

    void project(const Template &src, Template &dst) const
    {
        QList<QPointF> points = src.file.points();
        QList<QRectF> rects = src.file.rects();

        if (points.empty() || rects.empty()) {
            dst = src;
            if (Globals->verbose) qWarning("Delauney triangulation failed because points or rects are empty.");
            return;
        }

        int cols = src.m().cols;
        int rows = src.m().rows;

        // Assume rect appended last was bounding box
        points.append(rects.last().topLeft());
        points.append(rects.last().topRight());
        points.append(rects.last().bottomLeft());
        points.append(rects.last().bottomRight());

        Subdiv2D subdiv(Rect(0,0,cols,rows));
        // Make sure points are valid for Subdiv2D
        // TODO: Modify points to make them valid
        for (int i = 0; i < points.size(); i++) {
            if (points[i].x() < 0 || points[i].y() < 0 || points[i].y() >= rows || points[i].x() >= cols) {
                dst = src;
                if (Globals->verbose) qWarning("Delauney triangulation failed because points lie on boundary.");
                return;
            }
            subdiv.insert(OpenCVUtils::toPoint(points[i]));
        }

        vector<Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);

        QList<QPointF> validTriangles;

        for (size_t i = 0; i < triangleList.size(); i++) {
            // Check the triangle to make sure it's falls within the matrix
            bool valid = true;

            QList<QPointF> vertices;
            vertices.append(QPointF(triangleList[i][0],triangleList[i][1]));
            vertices.append(QPointF(triangleList[i][2],triangleList[i][3]));
            vertices.append(QPointF(triangleList[i][4],triangleList[i][5]));
            for (int j = 0; j < 3; j++) if (vertices[j].x() > cols || vertices[j].y() > rows || vertices[j].x() < 0 || vertices[j].y() < 0) valid = false;

            if (valid) validTriangles.append(vertices);
        }

        if (warp) {
            dst.m() = Mat::zeros(rows,cols,src.m().type());

            QList<float> procrustesStats = src.file.getList<float>("ProcrustesStats");

            Eigen::MatrixXf R(2,2);
            R(0,0) = procrustesStats.at(0);
            R(1,0) = procrustesStats.at(1);
            R(1,1) = procrustesStats.at(2);
            R(0,1) = procrustesStats.at(3);

            cv::Scalar mean(2);
            mean[0] = procrustesStats.at(4);
            mean[1] = procrustesStats.at(5);

            float norm = procrustesStats.at(6);

            QList<Point2f> mappedPoints;

            for (int i = 0; i < validTriangles.size(); i+=3) {
                // Matrix to store original (pre-transformed) triangle vertices
                Eigen::MatrixXf srcMat(3, 2);

                for (int j = 0; j < 3; j++) {
                    srcMat(j,0) = (validTriangles[i+j].x()-mean[0])/norm;
                    srcMat(j,1) = (validTriangles[i+j].y()-mean[1])/norm;
                }

                Eigen::MatrixXf dstMat = srcMat*R;

                Point2f srcPoints[3];
                for (int j = 0; j < 3; j++) srcPoints[j] = OpenCVUtils::toPoint(validTriangles[i+j]);

                Point2f dstPoints[3];
                for (int j = 0; j < 3; j++) {
                    // Scale and shift destination points
                    Point2f warpedPoint = Point2f(dstMat(j,0)*scaleFactor+cols/2,dstMat(j,1)*scaleFactor+rows/2);
                    dstPoints[j] = warpedPoint;
                    mappedPoints.append(warpedPoint);
                }

                Mat buffer(rows,cols,src.m().type());

                warpAffine(src.m(), buffer, getAffineTransform(srcPoints, dstPoints), Size(cols,rows));

                Mat mask = Mat::zeros(rows, cols, CV_8UC1);
                Point maskPoints[1][3];
                maskPoints[0][0] = dstPoints[0];
                maskPoints[0][1] = dstPoints[1];
                maskPoints[0][2] = dstPoints[2];
                const Point* ppt = { maskPoints[0] };

                fillConvexPoly(mask, ppt, 3, Scalar(255,255,255), 8);

                Mat output(rows,cols,src.m().type());

                if (i > 0) {
                    Mat overlap;
                    bitwise_and(dst.m(),mask,overlap);
                    mask.setTo(0, overlap!=0);
                }

                bitwise_and(buffer,mask,output);

                dst.m() += output;
            }

            // Overwrite any rects
            Rect boundingBox = boundingRect(mappedPoints.toVector().toStdVector());
            dst.file.setRects(QList<QRectF>() << OpenCVUtils::fromRect(boundingBox));
        } else dst = src;

        dst.file.setList<QPointF>("DelaunayTriangles", validTriangles);
    }
};

BR_REGISTER(Transform, DelaunayTransform)

/*!
 * \ingroup transforms
 * \brief  Maps texture from one set of points to another. Assumes that points are rigidly transformed
 * \author Brendan Klare \cite bklare
 * \author Scott Klum \cite sklum
 */
class TextureMapTransform : public UntrainableTransform
{
    Q_OBJECT

    static QRectF getBounds(QList<QPointF> points, int padding) {
        float srcMinX = FLT_MAX;
        float srcMinY = FLT_MAX;
        float srcMaxX = -FLT_MAX;
        float srcMaxY = -FLT_MAX;
        for (int i = 0; i < points.size(); i++) {
            if (points[i].x() < srcMinX)	srcMinX = points[i].x();
            if (points[i].y() < srcMinY)	srcMinY = points[i].y();
            if (points[i].x() > srcMaxX)	srcMaxX = points[i].x();
            if (points[i].y() > srcMaxY)	srcMaxY = points[i].y();
        }
        return QRectF(qRound(srcMinX - padding), qRound(srcMinY - padding), qRound(srcMaxX - srcMinX + 2 * padding), qRound(srcMaxY - srcMinY + 2 * padding));
    }

    static int getVertexIndex(QPointF trianglePts, QList<QPointF> pts) {
        for (int i = 0; i < pts.size(); i++)
            if (trianglePts.x() == pts[i].x() && trianglePts.y() == pts[i].y())
                return i;
        return -1;
    }

    QList<QList<int> > getTriangulation(const QList<QPointF> _points, const QRectF bound) const {
        QList<QPointF> points(_points);

        /*
        points.append(bound.topLeft());
        points.append(QPointF(bound.right() - 1, bound.top()));
        points.append(QPointF(bound.left(), bound.bottom() - 1));
        points.append(QPointF(bound.right() - 1, bound.bottom() - 1));
        points.append(QPointF(bound.left() + bound.width() / 2, bound.top()));
        points.append(QPointF(bound.left() + bound.width() / 2, bound.bottom() - 1));
        points.append(QPointF(bound.left(), bound.top() + bound.height() / 2));
        points.append(QPointF(bound.right() - 1, bound.top() + bound.height() / 2));
        */
        Subdiv2D subdiv(OpenCVUtils::toRect(bound));
        for (int i = 0; i < points.size(); i++)
            subdiv.insert(OpenCVUtils::toPoint(points[i]));

        vector<Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);

        QList<QList<int> > triangleIndices;
        for (size_t i = 0; i < triangleList.size(); i++) {
            bool valid = true;
            QList<QPointF> vertices;
            vertices.append(QPointF(triangleList[i][0],triangleList[i][1]));
            vertices.append(QPointF(triangleList[i][2],triangleList[i][3]));
            vertices.append(QPointF(triangleList[i][4],triangleList[i][5]));
            for (int j = 0; j < 3; j++)
                if (vertices[j].x() > bound.right() || vertices[j].y() > bound.bottom() || vertices[j].x() < bound.left() || vertices[j].y() < bound.top())
                    valid = false;

            if (valid) {
                QList<int> tri;
                for (int j = 0; j < 3; j++)
                    tri.append(getVertexIndex(vertices[j], points));
                triangleIndices.append(tri);
            }
        }

        return triangleIndices;
    }

    void project(const Template &src, Template &dst) const
    {
        QList<QPointF> dstPoints = dst.file.getList<QPointF>("ProcrustesPoints");
        QList<QPointF> srcPoints = dst.file.points();
        QRectF dstBound  = getBounds(dstPoints, 8);
        QRectF srcBound  = getBounds(srcPoints, 8);
        if (dstPoints.empty() || srcPoints.empty()) {
            dst = src;
            if (Globals->verbose) qWarning("Delauney triangulation failed because points or rects are empty.");
            return;
        }

        QList<QList<int> > triIndices = getTriangulation(srcPoints, srcBound);

        //QList<QPointF> dstTri = getTriangulation(dstPoints, dstBound);
        //QList<QPointF> srcTri = getTriangulation(srcPoints, srcBound);

        int dstWidth = dstBound.width() + dstBound.x();
        int dstHeight = dstBound.height() + dstBound.y();
        dst.m() = Mat::zeros(dstHeight, dstWidth, src.m().type());
static int SCNT = 0;
        for (int i = 0; i < triIndices.size(); i++) {
            Point2f srcPoint1[3];
            Point2f dstPoint1[3];
            for (int j = 0; j < 3; j++) {
                srcPoint1[j] = OpenCVUtils::toPoint(srcPoints[triIndices[i][j]]);
                dstPoint1[j] = OpenCVUtils::toPoint(dstPoints[triIndices[i][j]]);
            }

            Mat buffer(dstHeight, dstWidth, src.m().type());
            warpAffine(src.m(), buffer, getAffineTransform(srcPoint1, dstPoint1), Size(dstBound.width(), dstBound.height()));

            Mat mask = Mat::zeros(dstHeight, dstWidth, CV_8UC1);
            //Mat mask = Mat::zeros(dstHeight, dstWidth, src.m().type());
            Point maskPoints[1][3];
            maskPoints[0][0] = dstPoint1[0];
            maskPoints[0][1] = dstPoint1[1];
            maskPoints[0][2] = dstPoint1[2];
            const Point* ppt = { maskPoints[0] };
            fillConvexPoly(mask, ppt, 3, Scalar(255, 255, 255), 8);

            //dst.m().setTo(buffer, mask);
            //bitwise_and(buffer, mask, dst.m(), mask);
            for (int i = 0; i < dstHeight; i++) {
                for (int j = 0; j < dstWidth; j++) {
                    if (mask.at<uchar>(i,j) == 255) {
                        if (dst.m().type() == CV_32FC3 || dst.m().type() == CV_8UC3)
                            dst.m().at<cv::Vec3b>(i,j) = buffer.at<cv::Vec3b>(i,j);
                        else if (dst.m().type() == CV_32F)
                            dst.m().at<float>(i,j) = buffer.at<float>(i,j);
                        else if (dst.m().type() == CV_8U)
                            dst.m().at<uchar>(i,j) = buffer.at<uchar>(i,j);
                        else
                            qFatal("Unsupported pixel format.");
                    }
                        //dst.m().at<src.m().type()>i,j) = 255;
                }
            }

            //Mat output(dstBound.height(), dstBound.width(), src.m().type());

            /*
            if (i > 0) {
                Mat overlap;
                bitwise_and(dst.m(), mask, overlap);
                mask.setTo(0, overlap != 0);
            }
            */

            //bitwise_and(buffer,mask,output);
            //dst.m() += output;
        }
        /*
qDebug() <<  dst.m().rows << dst.m().cols << dstBound;
Eigen::Map<const Eigen::VectorXf> M(dst.m().ptr<float>(), dst.m().rows*dst.m().cols);
writeEigen((MatrixXf)M, QString("Temp/img%1.bin").arg(SCNT++));
OpenCVUtils::saveImage(dst.m(), QString("Temp/img%1.jpg").arg(SCNT));
*/
    }
};

BR_REGISTER(Transform, TextureMapTransform)

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

/*!
 * \ingroup transforms
 * \brief Read landmarks from a file and associate them with the correct templates.
 * \author Scott Klum \cite sklum
 *
 * Example of the format:
 * \code
 * image_001.jpg:146.000000,190.000000,227.000000,186.000000,202.000000,256.000000
 * image_002.jpg:75.000000,235.000000,140.000000,225.000000,91.000000,300.000000
 * image_003.jpg:158.000000,186.000000,246.000000,188.000000,208.000000,233.000000
 * \endcode
 */
class ReadLandmarksTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QString file READ get_file WRITE set_file RESET reset_file STORED false)
    Q_PROPERTY(QString imageDelimiter READ get_imageDelimiter WRITE set_imageDelimiter RESET reset_imageDelimiter STORED false)
    Q_PROPERTY(QString landmarkDelimiter READ get_landmarkDelimiter WRITE set_landmarkDelimiter RESET reset_landmarkDelimiter STORED false)
    BR_PROPERTY(QString, file, QString())
    BR_PROPERTY(QString, imageDelimiter, ":")
    BR_PROPERTY(QString, landmarkDelimiter, ",")

    QHash<QString, QList<QPointF> > landmarks;

    void init()
    {
        if (file.isEmpty())
            return;

        QFile f(file);
        if (!f.open(QFile::ReadOnly | QFile::Text))
            qFatal("Failed to open %s for reading.", qPrintable(f.fileName()));

        while (!f.atEnd()) {
            const QStringList words = QString(f.readLine()).split(imageDelimiter);
            const QStringList lm = words[1].split(landmarkDelimiter);

            QList<QPointF> points;
            bool ok;
            for (int i=0; i<lm.size(); i+=2)
                points.append(QPointF(lm[i].toFloat(&ok),lm[i+1].toFloat(&ok)));
            if (!ok) qFatal("Failed to read landmark.");

            landmarks.insert(words[0],points);
        }
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        dst.file.appendPoints(landmarks[dst.file.fileName()]);
    }
};

BR_REGISTER(Transform, ReadLandmarksTransform)

/*!
 * \ingroup transforms
 * \brief Name a point
 * \author Scott Klum \cite sklum
 */
class NamePointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    Q_PROPERTY(QStringList names READ get_names WRITE set_names RESET reset_names STORED false)
    BR_PROPERTY(QList<int>, indices, QList<int>())
    BR_PROPERTY(QStringList, names, QStringList())

    void projectMetadata(const File &src, File &dst) const
    {
        if (indices.size() != names.size()) qFatal("Point/name size mismatch");

        dst = src;

        QList<QPointF> points = src.points();

        for (int i=0; i<indices.size(); i++) {
            if (indices[i] < points.size()) dst.set(names[i], points[indices[i]]);
            else qFatal("Index out of range.");
        }
    }
};

BR_REGISTER(Transform, NamePointsTransform)

/*!
 * \ingroup transforms
 * \brief Remove a name from a point
 * \author Scott Klum \cite sklum
 */
class AnonymizePointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList names READ get_names WRITE set_names RESET reset_names STORED false)
    BR_PROPERTY(QStringList, names, QStringList())

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        foreach (const QString &name, names)
            if (src.contains(name)) dst.appendPoint(src.get<QPointF>(name));
    }
};

BR_REGISTER(Transform, AnonymizePointsTransform)

} // namespace br

#include "landmarks.moc"
