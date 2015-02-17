/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
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

#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>

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
class DelaunayTransform : public UntrainableTransform
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

} // namespace br

#include "metadata/delaunay.moc"
