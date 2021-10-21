/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2015 Rank One Computing Corporation                             *
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

namespace br
{

/*!
 * \ingroup transforms
 * \brief Checks the rects in a template for invalid values. The user can specify to fix the rects or remove them.
 * \author Keyur Patel \cite kpatel
 */

class CheckRectsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(bool fixRects READ get_fixRects WRITE set_fixRects RESET reset_fixRects STORED false)
    Q_PROPERTY(bool removeBadRect READ get_removeBadRect WRITE set_removeBadRect RESET reset_removeBadRect STORED false)
    BR_PROPERTY(bool, fixRects, false)
    BR_PROPERTY(bool, removeBadRect, true)

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.file.clearRects();
        QList<QRectF> rects = src.file.rects();

        if (fixRects) {
            foreach (QRectF r, rects)
                if ((r.height() <= src.m().rows) && (r.width() <= src.m().cols)) /* can't fix rects that don't fit the image */ {
                    if (r.left() < 0)                   r.moveLeft(0);
                    if (r.right() > src.m().cols-1)     r.moveRight(src.m().cols-1);
                    if (r.top() < 0)                    r.moveTop(0);
                    if (r.bottom() > src.m().rows-1)    r.moveBottom(src.m().rows-1);
                    dst.file.appendRect(r);
                }
        } else {
            foreach (QRectF r, rects){
                if ((r.left() < 0) || (r.right() > src.m().cols-1) || (r.top() < 0) || (r.bottom() > src.m().rows-1)){
                    if (removeBadRect){
                        rects.removeOne(r);
                    }
                    else {
                        dst.file.fte = true;
                        break;
                    }
                }
            }
            dst.file.setRects(rects);
        }
    }
};

BR_REGISTER(Transform, CheckRectsTransform)

} // namespace br

#include "metadata/checkrects.moc"

