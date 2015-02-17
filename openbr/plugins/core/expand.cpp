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
 * \brief Performs an expansion step on input templatelists
 * \author Josh Klontz \cite jklontz
 *
 * Each matrix in an input Template is expanded into its own template.
 *
 * \see PipeTransform
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
