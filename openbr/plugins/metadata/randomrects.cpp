/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2015 Rank One Computing Corporation                             *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Creates random rectangles within an image. Used for creating negative samples.
 * \author Brendan Klare \cite bklare
 */
class RandomRectsTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int numRects READ get_numRects WRITE set_numRects RESET reset_numRects STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    BR_PROPERTY(int, numRects, 135)
    BR_PROPERTY(int, minSize, 24)

    void project(const Template &src, Template &dst) const
    {
        qFatal("NOT SUPPORTED");
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach (const Template &t, src) {
            int maxSize = std::min(t.m().rows, t.m().cols);
            for (int i = 0; i < numRects; i++) {
                int size = (rand() % (maxSize - minSize)) + minSize;
                int x = rand() % (t.m().cols - size);
                int y = rand() % (t.m().rows - size);

                Template out(t.file, t.m());
                out.file.clearRects();
                out.file.appendRect(QRect(x,y,size,size));
                out.file.set("FrontalFace", QRect(x,y,size,size));
                dst.append(out);
            }
        }
    }
};

BR_REGISTER(Transform, RandomRectsTransform)

} // namespace br

#include "metadata/randomrects.moc"
