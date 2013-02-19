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
#include <openbr_plugin.h>

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
 * \brief Concatenates all input matrices into a single floating point matrix.
 * No requirements are placed on input matrices size and type.
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
            qFatal("Cat %d partitions does not evenly divide %d matrices.", partitions, src.size());
        QVector<int> sizes(partitions, 0);
        for (int i=0; i<src.size(); i++)
            sizes[i%partitions] += src[i].total() * src[i].channels();

        foreach (int size, sizes)
            dst.append(Mat(1, size, CV_32FC1));

        QVector<int> offsets(partitions, 0);
        for (int i=0; i<src.size(); i++) {
            size_t size = src[i].total() * src[i].elemSize();
            int j = i%partitions;
            memcpy(&dst[j].data[offsets[j]], src[i].ptr(), size);
            offsets[j] += size;
        }
    }
};

BR_REGISTER(Transform, CatTransform)

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

class RectFromLandmarksTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    Q_PROPERTY(double padding READ get_padding WRITE set_padding RESET reset_padding STORED false)
    Q_PROPERTY(double aspectRatio READ get_aspectRatio WRITE set_aspectRatio RESET reset_aspectRatio STORED false)
    BR_PROPERTY(QList<int>, indices, QList<int>())
    BR_PROPERTY(double, padding, 0)
    BR_PROPERTY(double, aspectRatio, 1.0)

    void project(const Template &src, Template &dst) const
    {
        if (src.file.landmarks().isEmpty()) qFatal("No landmarks");

        int minX, minY;
        minX = minY = std::numeric_limits<int>::max();
        int maxX, maxY;
        maxX = maxY = -std::numeric_limits<int>::max();

        foreach(int index, indices) {
            if (src.file.landmarks()[index].x() < minX) minX = src.file.landmarks()[index].x();
            if (src.file.landmarks()[index].x() > maxX) maxX = src.file.landmarks()[index].x();
            if (src.file.landmarks()[index].y() < minY) minY = src.file.landmarks()[index].y();
            if (src.file.landmarks()[index].y() > maxY) maxY = src.file.landmarks()[index].y();
            dst.file.appendLandmark(src.file.landmarks()[index]);
        }

        double width = maxX-minX+padding;
        double height = maxY-minY+padding;
        //double deltaHeight = ((width/aspectRatio) - height);

        dst.m() = src.m()(Rect(std::max(0.0, minX-padding/2.0), std::max(0.0, minY-padding/2.0), width, height));
    }
};

BR_REGISTER(Transform, RectFromLandmarksTransform)

} // namespace br

#include "regions.moc"
