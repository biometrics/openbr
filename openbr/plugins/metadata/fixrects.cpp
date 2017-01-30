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


/*!
 * \ingroup transforms
 * \brief Checks the rects in a template for invalid values
 * \author Keyur Patel \cite kpatel
 */

class CheckRectsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(bool removeBadRect READ get_removeBadRect WRITE set_removeBadRect RESET reset_removeBadRect STORED false)
    BR_PROPERTY(bool, removeBadRect, true)

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.file.clearRects();
        QList<QRectF> rects = src.file.rects();
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
};

BR_REGISTER(Transform, CheckRectsTransform)


} // namespace br

#include "metadata/fixrects.moc"
