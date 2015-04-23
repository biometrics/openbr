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

static TemplateList Expanded(const TemplateList &templates)
{
    TemplateList expanded;
    foreach (const Template &t, templates) {
        const bool enrollAll = t.file.get<bool>("enrollAll");
        if (t.isEmpty()) {
            if (!enrollAll)
                expanded.append(t);
            continue;
        }

        const QList<QPointF> points = t.file.points();
        const QList<QRectF> rects = t.file.rects();
        if (points.size() % t.size() != 0) qFatal("Uneven point count.");
        if (rects.size() % t.size() != 0) qFatal("Uneven rect count.");
        const int pointStep = points.size() / t.size();
        const int rectStep = rects.size() / t.size();

        for (int i=0; i<t.size(); i++) {
            expanded.append(Template(t.file, t[i]));
            expanded.last().file.setRects(rects.mid(i*rectStep, rectStep));
            expanded.last().file.setPoints(points.mid(i*pointStep, pointStep));
        }
    }
    return expanded;
}

/*!
 * \ingroup transforms
 * \brief Performs an expansion step on an input TemplateList. Each matrix in each input Template is expanded into its own Template.
 * \author Josh Klontz \cite jklontz
 * \br_related_plugin PipeTransform
 */
class ExpandTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    virtual void project(const TemplateList &src, TemplateList &dst) const
    {
        dst = Expanded(src);
    }

    virtual void project(const Template &src, Template &dst) const
    {
        dst = src;
        qDebug("Called Expand project(Template,Template), nothing will happen");
    }
};

BR_REGISTER(Transform, ExpandTransform)

} // namespace br

#include "core/expand.moc"
