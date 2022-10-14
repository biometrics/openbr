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

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Crops the rectangular regions of interest.
 * \author Josh Klontz \cite jklontz
 * \br_property QString propName Optional property name for a rectangle in metadata. If no propName is given the transform will use rects stored in the file.rects field.
 * \br_property bool copyOnCrop If true make a clone of each crop before appending the crop to dst. This guarantees that the crops will be continuous in memory, which is an occasionally useful property. Default is true.
 */
class ROITransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString propName READ get_propName WRITE set_propName RESET reset_propName STORED false)
    Q_PROPERTY(bool copyOnCrop READ get_copyOnCrop WRITE set_copyOnCrop RESET reset_copyOnCrop STORED false)
    Q_PROPERTY(int shiftPoints READ get_shiftPoints WRITE set_shiftPoints RESET reset_shiftPoints STORED false)
    BR_PROPERTY(QString, propName, "")
    BR_PROPERTY(bool, copyOnCrop, true)
    BR_PROPERTY(int, shiftPoints, -1)

    QRectF clipRect(const int img_width, const int img_height, const QRectF original) const
    {
        QRectF clipped;
        clipped.setX(max(0, (int)original.x()));
        clipped.setY(max(0, (int)original.y()));

        int width  = original.width()  + (original.x() < 0 ? original.x() : 0);
        int height = original.height() + (original.y() < 0 ? original.y() : 0);
        if (clipped.x() + width > img_width)
            width  -= ((clipped.x() + width)  - (img_width));
        if (clipped.y() + height > img_height)
            height -= ((clipped.y() + height) - (img_height));
        clipped.setWidth(width);
        clipped.setHeight(height);

        return clipped;
    }

    void project(const Template &src, Template &dst) const
    {
        QList<QRectF> rects;

        // check this first because the user specifically asked for this rect
        if (!propName.isEmpty()) {
            rects.append(clipRect(src.m().cols, src.m().rows, src.file.get<QRectF>(propName)));
        } else if ((propName == "Rects") || !src.file.rects().empty()) {
            foreach (const QRectF &rect, src.file.rects())
                rects.append(clipRect(src.m().cols, src.m().rows, rect));
        } else {
            dst = src;
            if (Globals->verbose)
                qWarning("No rects present in file.");
            return;
        }

        foreach (const QRectF &rect, rects)
            dst += src.m()(OpenCVUtils::toRect(rect));

        if (shiftPoints != -1) {
            // Shift the points to the rect with the index provided
            QRectF rect = src.file.rects()[shiftPoints];
            QList<QPointF> points = src.file.points();
            for (int i=0; i<points.size(); i++)
                points[i] -= rect.topLeft();
            dst.file.setPoints(points);
        }

        dst.file.clearRects();

        if (copyOnCrop)
            for (int i = 0; i < dst.size(); i++)
                dst.replace(i, dst[i].clone());
    }
};

BR_REGISTER(Transform, ROITransform)

} // namespace br

#include "imgproc/roi.moc"
