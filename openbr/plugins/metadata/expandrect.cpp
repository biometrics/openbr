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


namespace br
{

/*!
 * \ingroup transforms
 * \brief Expand the width and height of a Template's rects by input width and height factors.
 * \author Charles Otto \cite caotto
 * \author Brendan Klare \cite bklare
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
        dst.file.clearRects();
        QList<QRectF> rects = src.file.rects();
        for (int i=0;i < rects.size(); i++) {
            qreal widthGrowth = rects[i].width() * widthExpand;
            qreal heightGrowth = rects[i].height() * heightExpand;

            qreal x = std::max(qreal(0), rects[i].x() - widthGrowth / 2.);
            qreal y = std::max(qreal(0), rects[i].y() - heightGrowth / 2.);
            dst.file.appendRect(QRectF(x, y,
                             std::min(src.m().cols - x - 1, rects[i].width() + widthGrowth),
                             std::min(src.m().rows - y - 1, rects[i].height() + heightGrowth)));

        }
    }
};

BR_REGISTER(Transform, ExpandRectTransform)

} // namespace br

#include "metadata/expandrect.moc"
