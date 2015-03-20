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

#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Draws a grid on the image
 * \author Josh Klontz \cite jklontz
 */
class DrawGridLinesTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int rows READ get_rows WRITE set_rows RESET reset_rows STORED false)
    Q_PROPERTY(int columns READ get_columns WRITE set_columns RESET reset_columns STORED false)
    Q_PROPERTY(int r READ get_r WRITE set_r RESET reset_r STORED false)
    Q_PROPERTY(int g READ get_g WRITE set_g RESET reset_g STORED false)
    Q_PROPERTY(int b READ get_b WRITE set_b RESET reset_b STORED false)
    BR_PROPERTY(int, rows, 0)
    BR_PROPERTY(int, columns, 0)
    BR_PROPERTY(int, r, 196)
    BR_PROPERTY(int, g, 196)
    BR_PROPERTY(int, b, 196)

    void project(const Template &src, Template &dst) const
    {
        Mat m = src.m().clone();
        float rowStep = 1.f * m.rows / (rows+1);
        float columnStep = 1.f * m.cols / (columns+1);
        int thickness = qMin(m.rows, m.cols) / 256;
        for (float row = rowStep/2; row < m.rows; row += rowStep)
            line(m, Point(0, row), Point(m.cols, row), Scalar(r, g, b), thickness, CV_AA);
        for (float column = columnStep/2; column < m.cols; column += columnStep)
            line(m, Point(column, 0), Point(column, m.rows), Scalar(r, g, b), thickness, CV_AA);
        dst = m;
    }
};

BR_REGISTER(Transform, DrawGridLinesTransform)

} // namespace br

#include "gui/drawgridlines.moc"
