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

#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Subdivide matrix into rectangular subregions.
 * \author Josh Klontz \cite jklontz
 */
class RectRegionsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int width READ get_width WRITE set_width RESET reset_width STORED false)
    Q_PROPERTY(int height READ get_height WRITE set_height RESET reset_height STORED false)
    Q_PROPERTY(int widthStep READ get_widthStep WRITE set_widthStep RESET reset_widthStep STORED false)
    Q_PROPERTY(int heightStep READ get_heightStep WRITE set_heightStep RESET reset_heightStep STORED false)
    BR_PROPERTY(int, width, 8)
    BR_PROPERTY(int, height, 8)
    BR_PROPERTY(int, widthStep, -1)
    BR_PROPERTY(int, heightStep, -1)

    void project(const Template &src, Template &dst) const
    {
        const int widthStep = this->widthStep == -1 ? width : this->widthStep;
        const int heightStep = this->heightStep == -1 ? height : this->heightStep;
        const Mat &m = src;
        const int xMax = m.cols - width;
        const int yMax = m.rows - height;
        for (int x=0; x <= xMax; x += widthStep)
            for (int y=0; y <= yMax; y += heightStep)
                dst += m(Rect(x, y, width, height));
    }
};

BR_REGISTER(Transform, RectRegionsTransform)

/*!
 * \ingroup transforms
 * \brief Turns each row into its own matrix.
 * \author Josh Klontz \cite jklontz
 */
class ByRowTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        for (int i=0; i<src.m().rows; i++)
            dst += src.m().row(i);
    }
};

BR_REGISTER(Transform, ByRowTransform)

/*!
 * \ingroup transforms
 * \brief Concatenates all input matrices into a single matrix.
 * \author Josh Klontz \cite jklontz
 */
class CatTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int partitions READ get_partitions WRITE set_partitions RESET reset_partitions)
    BR_PROPERTY(int, partitions, 1)

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;

        if (src.size() % partitions != 0)
            qFatal("%d partitions does not evenly divide %d matrices.", partitions, src.size());
        QVector<int> sizes(partitions, 0);
        for (int i=0; i<src.size(); i++)
            sizes[i%partitions] += src[i].total();

        foreach (int size, sizes)
            dst.append(Mat(1, size, src.m().type()));

        QVector<int> offsets(partitions, 0);
        for (int i=0; i<src.size(); i++) {
            size_t size = src[i].total() * src[i].elemSize();
            int j = i % partitions;
            memcpy(&dst[j].data[offsets[j]], src[i].ptr(), size);
            offsets[j] += size;
        }
    }
};

BR_REGISTER(Transform, CatTransform)

/*!
 * \ingroup transforms
 * \brief Concatenates all input matrices by row into a single matrix.
 * All matricies must have the same row counts.
 * \author Josh Klontz \cite jklontz
 */
class CatRowsTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = Template(src.file, OpenCVUtils::toMatByRow(src));
    }
};

BR_REGISTER(Transform, CatRowsTransform)

/*!
 * \ingroup transforms
 * \brief Reshape the each matrix to the specified number of rows.
 * \author Josh Klontz \cite jklontz
 */
class ReshapeTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int rows READ get_rows WRITE set_rows RESET reset_rows STORED false)
    BR_PROPERTY(int, rows, 1)

    void project(const Template &src, Template &dst) const
    {
        dst = src.m().reshape(src.m().channels(), rows);
    }
};

BR_REGISTER(Transform, ReshapeTransform)

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV merge
 * \author Josh Klontz \cite jklontz
 */
class MergeTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        std::vector<Mat> mv;
        foreach (const Mat &m, src)
            mv.push_back(m);
        merge(mv, dst);
    }
};

BR_REGISTER(Transform, MergeTransform)

/*!
 * \ingroup transforms
 * \brief Duplicates the template data.
 * \author Josh Klontz \cite jklontz
 */
class DupTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    BR_PROPERTY(int, n, 1)

    void project(const Template &src, Template &dst) const
    {
        for (int i=0; i<n; i++)
            dst.merge(src);
    }
};

BR_REGISTER(Transform, DupTransform)

/*!
 * \ingroup transforms
 * \brief Create matrix from landmarks.
 * \author Scott Klum \cite sklum
 * \todo Padding should be a percent of total image size
 */

class RectFromPointsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    Q_PROPERTY(double padding READ get_padding WRITE set_padding RESET reset_padding STORED false)
    Q_PROPERTY(double aspectRatio READ get_aspectRatio WRITE set_aspectRatio RESET reset_aspectRatio STORED false)
    Q_PROPERTY(bool crop READ get_crop WRITE set_crop RESET reset_crop STORED false)
    BR_PROPERTY(QList<int>, indices, QList<int>())
    BR_PROPERTY(double, padding, 0)
    BR_PROPERTY(double, aspectRatio, 1.0)
    BR_PROPERTY(bool, crop, true)

    void project(const Template &src, Template &dst) const
    {
        if (src.file.points().isEmpty()) {
            qWarning("No landmarks");
            dst = src;
            return;
        }

        int minX, minY;
        minX = minY = std::numeric_limits<int>::max();
        int maxX, maxY;
        maxX = maxY = -std::numeric_limits<int>::max();

        QList<QPointF> points;

        foreach(int index, indices) {
            if (src.file.points().size() > index) {
                if (src.file.points()[index].x() < minX) minX = src.file.points()[index].x();
                if (src.file.points()[index].x() > maxX) maxX = src.file.points()[index].x();
                if (src.file.points()[index].y() < minY) minY = src.file.points()[index].y();
                if (src.file.points()[index].y() > maxY) maxY = src.file.points()[index].y();
                points.append(src.file.points()[index]);
            }
        }

        double width = maxX-minX;
        double deltaWidth = width*padding;
        width += deltaWidth;

        double height = maxY-minY;
        double deltaHeight = width/aspectRatio - height;
        height += deltaHeight;

        dst.file.setPoints(points);

        if (crop) dst.m() = src.m()(Rect(std::max(0.0, minX - deltaWidth/2.0), std::max(0.0, minY - deltaHeight/2.0), std::min((double)src.m().cols, width), std::min((double)src.m().rows, height)));
        else {
            dst.file.appendRect(QRectF(std::max(0.0, minX - deltaWidth/2.0), std::max(0.0, minY - deltaHeight/2.0), std::min((double)src.m().cols, width), std::min((double)src.m().rows, height)));
            dst.m() = src.m();
        }
    }
};

BR_REGISTER(Transform, RectFromPointsTransform)

} // namespace br

#include "regions.moc"
