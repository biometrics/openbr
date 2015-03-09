/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2015 Noblis                                                     *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <opencv2/opencv.hpp>
#include "openbr/plugins/openbr_internal.h"
#include "openbr/core/qtutils.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/eigenutils.h"
#include "openbr/core/common.h"
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
 * \brief Improved procrustes alignment of points, to include a post processing scaling of points
 * to faciliate subsequent texture mapping.
 * \author Brendan Klare \cite bklare
 * \param width Width of output coordinate space (before padding)
 * \param padding Amount of padding around the coordinate space
 * \param useFirst whether or not to use the first instance as the reference object
 */
class ProcrustesAlignTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(float width READ get_width WRITE set_width RESET reset_width STORED false)
    Q_PROPERTY(float padding READ get_padding WRITE set_padding RESET reset_padding STORED false)
    Q_PROPERTY(bool useFirst READ get_useFirst WRITE set_useFirst RESET reset_useFirst STORED false)
    BR_PROPERTY(float, width, 80)
    BR_PROPERTY(float, padding, 8)
    BR_PROPERTY(bool, useFirst, false)


    Eigen::MatrixXf referenceShape;
    float minX;
    float minY;
    float maxX;
    float maxY;
    float aspectRatio;

    void init() {
        aspectRatio = 0;
    }

    static MatrixXf getRotation(MatrixXf ref, MatrixXf sample) {
        MatrixXf R = sample.transpose() * ref;
        JacobiSVD<MatrixXf> svd(R, ComputeFullU | ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();
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
        int skip = 0;
        for (int j = 0; j < data.size(); j++) {
            QList<QPointF> imagePoints = data[j].file.points();
            if (imagePoints.size() == 0) {
                skip++;
                continue;
            }

            float meanX = 0,
                  meanY = 0;
            for (int i = 0; i < imagePoints.size(); i++) {
                points(i * 2, j - skip) = imagePoints[i].x();
                points(i * 2 + 1, j - skip) = imagePoints[i].y();

                meanX += imagePoints[i].x();
                meanY += imagePoints[i].y();
            }

            meanX /= imagePoints.size();
            meanY /= imagePoints.size();

            for (int i = 0; i < imagePoints.size(); i++) {
                points(i * 2, j - skip) -= meanX;
                points(i * 2 + 1, j - skip) -= meanY;
            }
        }
       
        points = MatrixXf(points.leftCols(data.size() - skip));
         
        //normalize scale
        for (int i = 0; i < points.cols(); i++)
            points.col(i) = points.col(i) / points.col(i).norm();

        //Normalize rotation
        if (!useFirst) {
            referenceShape = vectorToMatrix(points.rowwise().sum() / points.cols());
        } else {
            referenceShape = vectorToMatrix(points.col(0));
        }

        for (int i = 0; i < points.cols(); i++) {
            MatrixXf p = vectorToMatrix(points.col(i));
            MatrixXf R = getRotation(referenceShape, p);
            points.col(i) = matrixToVector(p * R);
        }

        //Choose crop boundaries and adjustments that captures most data
        MatrixXf minXs(points.cols(),1);
        MatrixXf minYs(points.cols(),1);
        MatrixXf maxXs(points.cols(),1);
        MatrixXf maxYs(points.cols(),1);
        for (int j = 0; j < points.cols(); j++) {
            minX = FLT_MAX,
            minY = FLT_MAX,
            maxX = -FLT_MAX,
            maxY = -FLT_MAX;
            for (int i = 0; i < points.rows(); i++) {
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

            minXs(j) = minX;
            maxXs(j) = maxX;
            minYs(j) = minY;
            maxYs(j) = maxY;
        }

        minX = minXs.mean() - 0 * EigenUtils::stddev(minXs);
        minY = minYs.mean() - 0 * EigenUtils::stddev(minYs);
        maxX = maxXs.mean() + 0 * EigenUtils::stddev(maxXs);
        maxY = maxYs.mean() + 0 * EigenUtils::stddev(maxYs);
        aspectRatio = (maxX - minX) / (maxY - minY);
    }

    void project(const Template &src, Template &dst) const
    {
        QList<QPointF> imagePoints = src.file.points();
        if (imagePoints.size() == 0) {
            dst.file.fte = true;
            qDebug() << "No points for file " << src.file.name;
            return;
        }

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
        p = p * R;

        //Translate and scale into output space and store in output list
        QList<QPointF> procrustesPoints;
        for (int i = 0; i < p.rows(); i++)
            procrustesPoints.append( QPointF(
                (p(i, 0) - minX) / (maxX - minX) * (width - 1) + padding,
                (p(i, 1) - minY) / (maxY - minY) * (qRound( width / aspectRatio) - 1) + padding));

        dst = src;
        dst.file.setList<QPointF>("ProcrustesPoints", procrustesPoints);
        dst.file.set("ProcrustesBound", QRectF(0, 0, width + 2 * padding, (qRound(width / aspectRatio) + 2 * padding)));
        dst.file.set("ProcrustesPadding", padding);
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
 * \brief  Maps texture from one set of points to another. Assumes that points are rigidly transformed
 * \author Brendan Klare \cite bklare
 * \author Scott Klum \cite sklum
 */
class TextureMapTransform : public UntrainableTransform
{
    Q_OBJECT

public:
    static QRectF getBounds(const QList<QPointF> &points, int dstPadding)
    {
        float srcMinX = FLT_MAX;
        float srcMinY = FLT_MAX;
        float srcMaxX = -FLT_MAX;
        float srcMaxY = -FLT_MAX;
        foreach (const QPointF &point, points) {
            if (point.x() < srcMinX) srcMinX = point.x();
            if (point.y() < srcMinY) srcMinY = point.y();
            if (point.x() > srcMaxX) srcMaxX = point.x();
            if (point.y() > srcMaxY) srcMaxY = point.y();
        }

        const float padding = (srcMaxX - srcMinX) / 80 * dstPadding;
        return QRectF(qRound(srcMinX - padding), qRound(srcMinY - padding), qRound(srcMaxX - srcMinX + 2 * padding), qRound(srcMaxY - srcMinY + 2 * padding));
    }

    static int getVertexIndex(const QPointF &trianglePts, const QList<QPointF> &pts)
    {
        for (int i = 0; i < pts.size(); i++)
            // Check points using single precision accuracy to avoid potential rounding error
            if ((float(trianglePts.x()) == float(pts[i].x())) && (float(trianglePts.y()) == float(pts[i].y())))
                return i;
        qFatal("Couldn't identify index of requested point!");
        return -1;
    }

    static QList<QPointF> addBounds(QList<QPointF> points, const QRectF &bound)
    {
        points.append(bound.topLeft());
        points.append(QPointF(bound.right() - 1, bound.top()));
        points.append(QPointF(bound.left(), bound.bottom() - 1));
        points.append(QPointF(bound.right() - 1, bound.bottom() - 1));
        return points;
    }

    static QList<QPointF> removeBounds(const QList<QPointF> &points)
    {
        return points.mid(0, points.size() - 4);
    }

    //Expand out bounds placed at end of point list by addBounds
    static QList<QPointF> expandBounds(QList<QPointF> points, int pad)
    {
        const int n = points.size();
        points[n-4] = QPointF(points[n-4].x() - pad, points[n-4].y() - pad);
        points[n-3] = QPointF(points[n-3].x() + pad, points[n-3].y() - pad);
        points[n-2] = QPointF(points[n-2].x() - pad, points[n-2].y() + pad);
        points[n-1] = QPointF(points[n-1].x() + pad, points[n-1].y() + pad);
        return points;
    }

    //Contract in bounds placed at end of point list by addBounds
    static QList<QPointF> contractBounds(QList<QPointF> points, int pad)
    {
        const int n = points.size();
        points[n-4] = QPointF(points[n-4].x() + pad, points[n-4].y() + pad);
        points[n-3] = QPointF(points[n-3].x() - pad, points[n-3].y() + pad);
        points[n-2] = QPointF(points[n-2].x() + pad, points[n-2].y() - pad);
        points[n-1] = QPointF(points[n-1].x() - pad, points[n-1].y() - pad);
        return points;
    }

    static QList<QList<int> > getTriangulation(const QList<QPointF> &points, const QRectF &bound)
    {
        Subdiv2D subdiv(OpenCVUtils::toRect(bound));
        foreach (const QPointF &point, points) {
            if (!bound.contains(point))  
                return QList<QList<int> >();
            subdiv.insert(OpenCVUtils::toPoint(point));
        }


        vector<Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);

        QList<QList<int> > triangleIndices;
        foreach (const Vec6f &triangle, triangleList) {
            bool valid = true;
            const QPointF vertices[3] = { QPointF(triangle[0], triangle[1]),
                                          QPointF(triangle[2], triangle[3]),
                                          QPointF(triangle[4], triangle[5]) };
            for (int j = 0; j < 3; j++)
                if (vertices[j].x() > bound.right() || vertices[j].y() > bound.bottom() || vertices[j].x() < bound.left() || vertices[j].y() < bound.top()) {
                    valid = false;
                    break;
                }

            if (valid) {
                QList<int> tri;
                for (int j = 0; j < 3; j++)
                    tri.append(getVertexIndex(vertices[j], points));
                triangleIndices.append(tri);
            }
        }

        return triangleIndices;
    }

private:
    void project(const Template &src, Template &dst) const
    {
        QList<QPointF> dstPoints = dst.file.getList<QPointF>("ProcrustesPoints");
        QList<QPointF> srcPoints = dst.file.points();
        if (dstPoints.empty() || srcPoints.empty()) {
            dst = src;
            if (Globals->verbose) {
                qWarning("Delauney triangulation failed because points or rects are empty.");
                dst.file.fte = true;
            }
            return;
        }

        QRectF dstBound  = dst.file.get<QRectF>("ProcrustesBound");
        dstPoints = addBounds(dstPoints, dstBound);

        /*Add a wider bound for triangulation to prevent border triangles from being missing*/
        QRectF srcBound  = getBounds(srcPoints, dst.file.get<int>("ProcrustesPadding") + 20);
        srcPoints = addBounds(srcPoints, srcBound);
        QList<QList<int> > triIndices = getTriangulation(srcPoints, srcBound);

        /*Remove wider bound for texture mapping*/
        srcPoints = removeBounds(srcPoints);
        srcBound  = getBounds(srcPoints, dst.file.get<int>("ProcrustesPadding"));
        srcPoints = addBounds(srcPoints, srcBound);

        int dstWidth = dstBound.width() + dstBound.x();
        int dstHeight = dstBound.height() + dstBound.y();
        dst.m() = Mat::zeros(dstHeight, dstWidth, src.m().type());
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
            Point maskPoints[1][3];
            maskPoints[0][0] = dstPoint1[0];
            maskPoints[0][1] = dstPoint1[1];
            maskPoints[0][2] = dstPoint1[2];
            const Point* ppt = { maskPoints[0] };
            fillConvexPoly(mask, ppt, 3, Scalar(255, 255, 255), 8);

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
                }
            }

        }

        dst.file = src.file;
        dst.file.clearPoints();
        dst.file.clearRects();
        dst.file.remove("ProcrustesPoints");
        dst.file.remove("ProcrustesPadding");
        dst.file.remove("ProcrustesBounds");
    }
};

BR_REGISTER(Transform, TextureMapTransform)

// SynthesizePointsTransform helper class
struct TriangleIndicies
{
    int indicies[3];

    TriangleIndicies()
    {
        indicies[0] = 0;
        indicies[1] = 0;
        indicies[2] = 0;
    }

    TriangleIndicies(QList<int> indexList)
    {
        assert(indexList.size() == 3);
        qSort(indexList);
        indicies[0] = indexList[0];
        indicies[1] = indexList[1];
        indicies[2] = indexList[2];
    }
};

inline bool operator==(const TriangleIndicies &a, const TriangleIndicies &b)
{
    return (a.indicies[0] == b.indicies[0]) && (a.indicies[1] == b.indicies[1]) && (a.indicies[2] == b.indicies[2]);
}

inline uint qHash(const TriangleIndicies &key)
{
    return ::qHash(key.indicies[0]) ^ ::qHash(key.indicies[1]) ^ ::qHash(key.indicies[2]);
}

QDataStream &operator<<(QDataStream &stream, const TriangleIndicies &ti)
{
    return stream << ti.indicies[0] << ti.indicies[1] << ti.indicies[2];
}

QDataStream &operator>>(QDataStream &stream, TriangleIndicies &ti)
{
    return stream >> ti.indicies[0] >> ti.indicies[1] >> ti.indicies[2];
}

/*!
 * \ingroup transforms
 * \brief Synthesize additional points via triangulation.
 * \author Josh Klontz \cite jklontz
 */
 class SynthesizePointsTransform : public MetadataTransform
 {
    Q_OBJECT
    Q_PROPERTY(float minRelativeDistance READ get_minRelativeDistance WRITE set_minRelativeDistance RESET reset_minRelativeDistance STORED false)
    BR_PROPERTY(float, minRelativeDistance, 0) // [0, 1] range controlling whether or not to nearby synthetic points.
                                               // 0 = keep all points, 1 = keep only the most distance point.

    QList<TriangleIndicies> triangles;

    void train(const TemplateList &data)
    {
        // Because not all triangulations are the same, we have to decide on a canonical set of triangles at training time.
        QHash<TriangleIndicies, int> counts;
        foreach (const Template &datum, data) {

            const QList<QPointF> points = datum.file.points();
            if (points.size() == 0)
                    continue;
            const QList< QList<int> > triangulation = TextureMapTransform::getTriangulation(points, TextureMapTransform::getBounds(points, 10));
            if (triangulation.empty())
                continue;
            
            foreach (const QList<int> &indicies, triangulation)
                counts[TriangleIndicies(indicies)]++;
        }

        triangles.clear();
        QHash<TriangleIndicies, int>::const_iterator i = counts.constBegin();
        while (i != counts.constEnd()) {
            if (3 * i.value() > data.size())
                triangles.append(i.key()); // Keep triangles that occur in at least a third of the training instances
            ++i;
        }

        if (minRelativeDistance > 0) { // Discard relatively small triangles
            QVector<float> averageMinDistances(triangles.size());
            foreach (const Template &datum, data) {
                File dst;
                projectMetadata(datum.file, dst);
                const QList<QPointF> points = dst.points();

                QVector<float> minDistances(triangles.size());
                for (int i=0; i<triangles.size(); i++) {
                    // Work backwards so that we can also consider distances between synthetic points
                    const QPointF &point = points[points.size()-1-i];
                    float minDistance = std::numeric_limits<float>::max();
                    for (int j=0; j<points.size()-1-i; j++)
                        minDistance = min(minDistance, sqrtf(powf(point.x() - points[j].x(), 2.f) + powf(point.y() - points[j].y(), 2.f)));
                    minDistances[triangles.size()-1-i] = minDistance;
                }

                const float maxMinDistance = Common::Max(minDistances);
                for (int i=0; i<minDistances.size(); i++)
                    averageMinDistances[i] += (minDistances[i] / maxMinDistance);
            }

            const float maxAverageMinDistance = Common::Max(averageMinDistances);
            for (int i=averageMinDistances.size()-1; i>=0; i--)
                if (averageMinDistances[i] / maxAverageMinDistance < minRelativeDistance)
                    triangles.removeAt(i);
        }

        if (Globals->verbose)
            qDebug() << "Kept" << triangles.size() << "of" << counts.size() << "triangles.";
    }

    void projectMetadata(const File &src, File &dst) const
    {
        QList<QPointF> points = src.points();
        if (points.size() == 0) {
            dst.fte = true;
            return;
        }

        foreach (const TriangleIndicies &triangle, triangles) {
            const QPointF &p0 = points[triangle.indicies[0]];
            const QPointF &p1 = points[triangle.indicies[1]];
            const QPointF &p2 = points[triangle.indicies[2]];
            points.append((p0 + p1 + p2) / 3 /* append the centroid of the triangle */);
        }
        dst.setPoints(points);
    }

    void store(QDataStream &stream) const
    {
        stream << triangles;
    }

    void load(QDataStream &stream)
    {
        stream >> triangles;
    }
 };
 BR_REGISTER(Transform, SynthesizePointsTransform)

/*!
 * \ingroup initializers
 * \brief Initialize Procrustes croppings
 * \author Brendan Klare \cite bklare
 */
class ProcrustesInitializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        Globals->abbreviations.insert("ProcrustesStasmFace","SelectPoints([17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76])+ProcrustesAlign(padding=6,width=120)+TextureMap+Resize(96,96)");
        Globals->abbreviations.insert("ProcrustesStasmEyes","SelectPoints([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47])+ProcrustesAlign(padding=8)+TextureMap+Resize(24,48)");
        Globals->abbreviations.insert("ProcrustesStasmPeriocular","SelectPoints([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,16,17,18,19,20,21,22,23,24,25,26,27])+ProcrustesAlign(padding=10)+TextureMap+Resize(36,48)");
        Globals->abbreviations.insert("ProcrustesStasmBrow","SelectPoints([16,17,18,19,20,21,22,23,24,25,26,27])+ProcrustesAlign(padding=8)+TextureMap+Resize(24,48)");
        Globals->abbreviations.insert("ProcrustesStasmNose","SelectPoints([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])+ProcrustesAlign(padding=12)+TextureMap+Resize(36,48)");
        Globals->abbreviations.insert("ProcrustesStasmMouth","SelectPoints([59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76])+ProcrustesAlign(padding=10)+TextureMap+Resize(36,48)");
        Globals->abbreviations.insert("ProcrustesStasmJaw", "SelectPoints([2,3,4,5,6,7,8,9,10])+ProcrustesAlign(padding=8)+TextureMap+Resize(36,48)");

        Globals->abbreviations.insert("ProcrustesEyes","SelectPoints([19,20,21,22,23,24,25,26,27,28,29,30])+ProcrustesAlign(padding=8)+TextureMap+Resize(24,48)");
        Globals->abbreviations.insert("ProcrustesNose","SelectPoints([12,13,14,15,16,17,18])+ProcrustesAlign(padding=30)+TextureMap+Resize(36,48)");
        Globals->abbreviations.insert("ProcrustesMouth","SelectPoints([31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50])+ProcrustesAlign(padding=6)+TextureMap+Resize(36,48)");
        Globals->abbreviations.insert("ProcrustesBrow","SelectPoints([0,1,2,3,4,5,6,7,8,9])+ProcrustesAlign(padding=6)+TextureMap+Resize(24,48)");
        Globals->abbreviations.insert("ProcrustesFace","ProcrustesAlign(padding=6,width=120)+TextureMap+Resize(96,96)");

        Globals->abbreviations.insert("ProcrustesLargeStasmFace","ProcrustesAlign(padding=18)+TextureMap+Resize(480,480)");
        Globals->abbreviations.insert("ProcrustesLargeStasmEyes","SelectPoints([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47])+ProcrustesAlign(padding=8)+TextureMap+Resize(240,480)");
        Globals->abbreviations.insert("ProcrustesLargeStasmPeriocular","SelectPoints([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,16,17,18,19,20,21,22,23,24,25,26,27])+ProcrustesAlign(padding=10)+TextureMap+Resize(360,480)");
        Globals->abbreviations.insert("ProcrustesLargeStasmBrow","SelectPoints([16,17,18,19,20,21,22,23,24,25,26,27])+ProcrustesAlign(padding=8)+TextureMap+Resize(240,480)");
        Globals->abbreviations.insert("ProcrustesLargeStasmNose","SelectPoints([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])+ProcrustesAlign(padding=12)+TextureMap+Resize(360,480)");
        Globals->abbreviations.insert("ProcrustesLargeStasmMouth","SelectPoints([59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76])+ProcrustesAlign(padding=20)+TextureMap+Resize(360,480)");
        Globals->abbreviations.insert("ProcrustesLargeStasmJaw", "SelectPoints([2,3,4,5,6,7,8,9,10])+ProcrustesAlign(padding=8)+TextureMap+Resize(360,480)");
    }
};
BR_REGISTER(Initializer, ProcrustesInitializer)

} // namespace br

#include "align.moc"
