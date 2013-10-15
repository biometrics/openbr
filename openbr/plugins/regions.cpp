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
#include "openbr/core/qtutils.h"
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

/*!
 * \ingroup transforms
 * \brief Subdivide matrix into a fixed number of rectangular subregions.
 * \author Brendan Klare \cite bklare
 */
class FixedRegionsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int nHorizontal READ get_nHorizontal WRITE set_nHorizontal RESET reset_nHorizontal STORED false)
    Q_PROPERTY(int nVertical READ get_nVertical WRITE set_nVertical RESET reset_nVertical STORED false)
    Q_PROPERTY(float widthScaleStep READ get_widthScaleStep WRITE set_widthScaleStep RESET reset_widthScaleStep STORED false)
    Q_PROPERTY(float heightScaleStep READ get_heightScaleStep WRITE set_heightScaleStep RESET reset_heightScaleStep STORED false)
    BR_PROPERTY(int, nHorizontal, 5)
    BR_PROPERTY(int, nVertical, 5)
    BR_PROPERTY(float, widthScaleStep, .5)
    BR_PROPERTY(float, heightScaleStep, .5)

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

        if (!src.empty())
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
 * All matricies must have the same column counts.
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
 * \brief Concatenates all input matrices by column into a single matrix.
 * Use after a fork to concatenate two feature matrices by column.
 * \author Austin Blanton \cite imaus10
 */
class CatColsTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        int half = src.size()/2;
        for (int i=0; i<half; i++) {
            Mat first = src[i];
            Mat second = src[half+i];
            Mat both;
            hconcat(first, second, both);
            dst.append(both);
        }
        dst.file = src.file;
    }
};

BR_REGISTER(Transform, CatColsTransform)

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
 * \brief Expand the width and height of a template's rects by input width and height factors.
 * \author Charles Otto \cite caotto
 */
class ExpandRectTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float widthExpand READ get_widthExpand WRITE set_widthExpand RESET reset_widthExpand STORED false)
    Q_PROPERTY(float heightExpand READ get_heightExpand WRITE set_heightExpand RESET reset_heightExpand STORED false)
    BR_PROPERTY(float, widthExpand, .5)
    BR_PROPERTY(float, heightExpand, .5)
    void project(const Template &src, Template &dst) const
    {
        dst = src;
        QList<QRectF> rects = dst.file.rects();
        for (int i=0;i < rects.size(); i++) {
            QRectF rect = rects[i];

            qreal width = rect.width();
            qreal height = rect.height();
            float half_w_expansion = widthExpand / 2;
            float half_h_expansion = heightExpand / 2;

            qreal half_width = width * widthExpand;
            qreal quarter_width = width * half_w_expansion;
            qreal half_height = height * heightExpand;
            qreal quarter_height = height * half_h_expansion;

            rect.setX(std::max(qreal(0),(rect.x() - quarter_width)));
            rect.setY(std::max(qreal(0),(rect.y() - quarter_height)));

            qreal x2 = std::min(rect.width() + half_width + rect.x(), qreal(src.m().cols) - 1);
            qreal y2 = std::min(rect.height() + half_height + rect.y(), qreal(src.m().rows) - 1);

            rect.setWidth(x2 - rect.x());
            rect.setHeight(y2 - rect.y());

            rects[i] = rect;
        }
        dst.file.setRects(rects);
    }
};

BR_REGISTER(Transform, ExpandRectTransform)

/*!
 * \ingroup transforms
 * \brief Crops the width and height of a template's rects by input width and height factors.
 * \author Scott Klum \cite sklum
 */
class CropRectTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QString widthCrop READ get_widthCrop WRITE set_widthCrop RESET reset_widthCrop STORED false)
    Q_PROPERTY(QString heightCrop READ get_heightCrop WRITE set_heightCrop RESET reset_heightCrop STORED false)
    BR_PROPERTY(QString, widthCrop, QString())
    BR_PROPERTY(QString, heightCrop, QString())

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        QList<QRectF> rects = src.file.rects();
        for (int i=0;i < rects.size(); i++) {
            rects[i].setX(rects[i].x() + rects[i].width() * QtUtils::toPoint(widthCrop).x());
            rects[i].setY(rects[i].y() + rects[i].height() * QtUtils::toPoint(heightCrop).x());
            rects[i].setWidth(rects[i].width() * (1-QtUtils::toPoint(widthCrop).y()));
            rects[i].setHeight(rects[i].height() * (1-QtUtils::toPoint(heightCrop).y()));
        }
        dst.file.setRects(rects);
    }
};

BR_REGISTER(Transform, CropRectTransform)

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
            else qFatal("Incorrect indices");
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
