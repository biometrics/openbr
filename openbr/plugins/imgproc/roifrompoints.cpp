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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Crops the rectangular regions of interest from given points and sizes.
 * \author Austin Blanton \cite imaus10
 */
class ROIFromPtsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int width READ get_width WRITE set_width RESET reset_width STORED false)
    Q_PROPERTY(int height READ get_height WRITE set_height RESET reset_height STORED false)
    BR_PROPERTY(int, width, 1)
    BR_PROPERTY(int, height, 1)

    void project(const Template &src, Template &dst) const
    {
        foreach (const QPointF &pt, src.file.points()) {
            int x = pt.x() - (width/2);
            int y = pt.y() - (height/2);
            dst += src.m()(Rect(x, y, width, height));
        }
    }
};

BR_REGISTER(Transform, ROIFromPtsTransform)

} // namespace br

#include "imgproc/roifrompoints.moc"
