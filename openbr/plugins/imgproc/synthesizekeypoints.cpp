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

    void train(const TemplateList &data)
    {
        // Because not all triangulations are the same, we have to decide on a canonical set of triangles at training time.
        QHash<TriangleIndicies, int> counts;
        foreach (const Template &datum, data) {
            const QList<QPointF> points = datum.file.points();
            if (points.size() <= 4)
                continue;

            const QList< QList<int> > triangulation = getTriangulation(points, getBounds(points, 10));
            if (triangulation.empty())
                continue;
            
            foreach (const QList<int> &indicies, triangulation)
                counts[TriangleIndicies(indicies)]++;
        }

        if (counts.empty())
            return;

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

} // namespace br

#include "synthesizekeypoints.moc"
