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

#include "core/opencvutils.h"

using namespace cv;
using namespace br;

/*!
 * \ingroup transforms
 * \brief Crops the regions of interest.
 * \author Josh Klontz \cite jklontz
 */
class ROI : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        foreach (const QRectF ROI, src.file.ROIs())
            dst += src.m()(OpenCVUtils::toRect(ROI));
    }
};

BR_REGISTER(Transform, ROI)

/*!
 * \ingroup transforms
 * \brief Resize the template
 * \author Josh Klontz \cite jklontz
 */
class Resize : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int rows READ get_rows WRITE set_rows RESET reset_rows STORED false)
    Q_PROPERTY(int columns READ get_columns WRITE set_columns RESET reset_columns STORED false)
    BR_PROPERTY(int, rows, -1)
    BR_PROPERTY(int, columns, -1)

    void project(const Template &src, Template &dst) const
    {
        resize(src, dst, Size((columns == -1) ? src.m().cols*rows/src.m().rows : columns, rows));
    }
};

BR_REGISTER(Transform, Resize)

/*!
 * \ingroup transforms
 * \brief Limit the size of the template
 * \author Josh Klontz \cite jklontz
 */
class LimitSize : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int max READ get_max WRITE set_max RESET reset_max STORED false)
    BR_PROPERTY(int, max, -1)

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        if (m.rows > m.cols)
            if (m.rows > max) resize(m, dst, Size(std::max(1, m.cols * max / m.rows), max));
            else              dst = m;
        else
            if (m.cols > max) resize(m, dst, Size(max, std::max(1, m.rows * max / m.cols)));
            else              dst = m;
    }
};

BR_REGISTER(Transform, LimitSize)

/*!
 * \ingroup transforms
 * \brief Crop out black borders
 * \author Josh Klontz \cite jklontz
 */
class CropBlack : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        Mat gray;
        OpenCVUtils::cvtGray(src, gray);

        int xStart = 0;
        while (xStart < gray.cols) {
            if (mean(gray.col(xStart))[0] >= 1) break;
            xStart++;
        }

        int xEnd = gray.cols - 1;
        while (xEnd >= 0) {
            if (mean(gray.col(xEnd))[0] >= 1) break;
            xEnd--;
        }

        int yStart = 0;
        while (yStart < gray.rows) {
            if (mean(gray.col(yStart))[0] >= 1) break;
            yStart++;
        }

        int yEnd = gray.rows - 1;
        while (yEnd >= 0) {
            if (mean(gray.col(yEnd))[0] >= 1) break;
            yEnd--;
        }

        dst = src.m()(Rect(xStart, yStart, xEnd-xStart, yEnd-yStart));
    }
};

BR_REGISTER(Transform, CropBlack)

#include "crop.moc"
