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
 * \brief Checks the rects in a template for invalid values
 * \author Keyur Patel \cite kpatel
 */

class CheckRectsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(bool removeBadRect READ get_removeBadRect WRITE set_removeBadRect RESET reset_removeBadRect STORED false)
    BR_PROPERTY(bool, removeBadRect, false)

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.file.clearRects();
        QList<QRectF> rects = src.file.rects();
        for (int i=0; i<rects.size(); i++) {
            QRectF r = rects[i];
            if ((r.left() < 0) || (r.right() > src.m().cols-1) || (r.top() < 0) || (r.bottom() > src.m().rows-1)){
                if (removeBadRect)
                    continue;
                else {
                    dst.file.fte = true;
                    break;
                }
            } else 
                dst.file.appendRect(r);

        }
    }
};

BR_REGISTER(Transform, CheckRectsTransform)

} // namespace br

#include "metadata/checkrects.moc"

