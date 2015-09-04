#include <openbr/plugins/openbr_internal.h>

namespace br
{

class FixRectsTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.file.clearRects();
        QList<QRectF> rects = src.file.rects();
        for (int i=0; i<rects.size(); i++) {
            QRectF r = rects[i];
            if (r.left() < 0) r.setLeft(0);
            if (r.right() > src.m().cols-1) r.setRight(src.m().cols-1);
            if (r.top() < 0) r.setTop(0);
            if (r.bottom() > src.m().rows-1) r.setBottom(src.m().rows-1);
            dst.file.appendRect(r);
        }
    }
};

BR_REGISTER(Transform, FixRectsTransform)

} // namespace br

#include "metadata/fixrects.moc"
