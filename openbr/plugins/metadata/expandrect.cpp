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
 * \brief Expand the width and height of a template's rects by input width and height factors.
 * \author Charles Otto \cite caotto
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
        QList<QRectF> rects = dst.file.rects();
        for (int i=0;i < rects.size(); i++) {
            QRectF rect = rects[i];

            qreal width = rect.width();
            qreal height = rect.height();
            float half_w_expansion = widthExpand / 2;
            float half_h_expansion = heightExpand / 2;

            qreal half_width = width * widthExpand;
            qreal quarter_width = width * half_w_expansion;
            qreal half_height = height * heightExpand;
            qreal quarter_height = height * half_h_expansion;

            rect.setX(std::max(qreal(0),(rect.x() - quarter_width)));
            rect.setY(std::max(qreal(0),(rect.y() - quarter_height)));

            qreal x2 = std::min(rect.width() + half_width + rect.x(), qreal(src.m().cols) - 1);
            qreal y2 = std::min(rect.height() + half_height + rect.y(), qreal(src.m().rows) - 1);

            rect.setWidth(x2 - rect.x());
            rect.setHeight(y2 - rect.y());

            rects[i] = rect;
        }
        dst.file.setRects(rects);
    }
};

BR_REGISTER(Transform, ExpandRectTransform)

} // namespace br

#include "metadata/expandrect.moc"
