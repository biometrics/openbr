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
 * \brief Scales a rectangle either by pixel value or factor.
 * \author Scott Klum \cite sklum
 * \br_property int index Index of rect to scale.
 * \br_property int left Number of pixels to scale the left of the rect.
 * \br_property int top Number of pixels to scale the top of the rect.
 * \br_property int right Number of pixels to scale the right of the rect.
 * \br_property int bottom Number of pixels to scale the bottom of the rect.
 * \br_property float factor Scale factor.
 */
class ScaleRectsTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(int index READ get_index WRITE set_index RESET reset_index STORED false)
    Q_PROPERTY(int left READ get_left WRITE set_left RESET reset_left STORED false)
    Q_PROPERTY(int top READ get_top WRITE set_top RESET reset_top STORED false)
    Q_PROPERTY(int right READ get_right WRITE set_right RESET reset_right STORED false)
    Q_PROPERTY(int bottom READ get_bottom WRITE set_bottom RESET reset_bottom STORED false)
    Q_PROPERTY(float factor READ get_factor WRITE set_factor RESET reset_factor STORED false)
    BR_PROPERTY(int, index, 0)
    BR_PROPERTY(int, left, -15)
    BR_PROPERTY(int, top, -25)
    BR_PROPERTY(int, right, 15)
    BR_PROPERTY(int, bottom, 5)
    BR_PROPERTY(float, factor, 0)

    void project(const Template &src, Template &dst) const {
        dst = src;
        QList<QRectF> rects = src.file.rects();
        if (factor) {
            QRectF orig = rects[index];
            float scaledArea = factor*orig.width()*orig.height();
            float scaledWidth = sqrt(scaledArea*(orig.width()/orig.height()));
            QRectF scaled(0, 0, scaledWidth, scaledWidth/(orig.width()/orig.height()));
            scaled.moveCenter(orig.center());
            rects.replace(index, scaled);
            dst.file.setRects(OpenCVUtils::toRects(rects));
            return;
        }

        QRectF rect = rects[index].adjusted(left, top, right, bottom);
        rects.replace(index, rect);
        dst.file.setRects(OpenCVUtils::toRects(rects));
    }
};

BR_REGISTER(Transform, ScaleRectsTransform)

} // namespace br

#include "metadata/scalerects.moc"
