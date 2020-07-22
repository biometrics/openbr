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
#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Add landmarks to the Template in a grid layout
 * \author Josh Klontz \cite jklontz
 */
class GridTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int rows READ get_rows WRITE set_rows RESET reset_rows STORED false)
    Q_PROPERTY(int columns READ get_columns WRITE set_columns RESET reset_columns STORED false)
    Q_PROPERTY(float border READ get_border WRITE set_border RESET reset_border STORED false)
    Q_PROPERTY(float angle READ get_angle WRITE set_angle RESET reset_angle STORED false)
    BR_PROPERTY(int, rows, 1)
    BR_PROPERTY(int, columns, 1)
    BR_PROPERTY(float, border, 0)
    BR_PROPERTY(float, angle, 0)

    void project(const Template &src, Template &dst) const
    {
        QList<QPointF> landmarks;
        const float row_border = (border < 1 ? src.m().rows*border : border);
        const float col_border = (border < 1 ? src.m().cols*border : border);
        const float row_step = (src.m().rows-row_border*2) / rows;
        const float column_step = (src.m().cols-col_border*2) / columns;
        for (float y=row_step/2+row_border; y<src.m().rows-row_border; y+=row_step)
            for (float x=column_step/2+col_border; x<src.m().cols-col_border; x+=column_step)
                landmarks.append(QPointF(x,y));

        if (angle > 0) {
            const Mat rotMatrix = getRotationMatrix2D(Point2f(src.m().cols / 2, src.m().rows / 2), angle, 1.0);
            landmarks = OpenCVUtils::rotatePoints(landmarks, rotMatrix);
        }

        dst = src;
        dst.file.setPoints(landmarks);
    }
};

BR_REGISTER(Transform, GridTransform)

} // namespace br

#include "metadata/grid.moc"
