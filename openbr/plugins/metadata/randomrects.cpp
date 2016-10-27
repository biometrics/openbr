/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2015 Rank One Computing Corporation                             *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

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
    Q_PROPERTY(bool sameSize READ get_sameSize WRITE set_sameSize RESET reset_sameSize STORED false)
    Q_PROPERTY(bool filterDarkPatches READ get_filterDarkPatches WRITE set_filterDarkPatches RESET reset_filterDarkPatches STORED false)
    Q_PROPERTY(float bgFgThresh READ get_bgFgThresh WRITE set_bgFgThresh RESET reset_bgFgThresh STORED false)

    BR_PROPERTY(int, numRects, 135)
    BR_PROPERTY(int, minSize, 24)
    BR_PROPERTY(bool, sameSize, false)
    BR_PROPERTY(bool, filterDarkPatches, false)
    BR_PROPERTY(float, bgFgThresh, 0.75)

    void project(const Template &, Template &) const
    {
        qFatal("NOT SUPPORTED");
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        int size = minSize;
        foreach (const Template &t, src) {
            int maxSize = std::min(t.m().rows, t.m().cols);
            for (int i = 0; i < numRects; i++) {

                if (!sameSize){
                     size = (rand() % (maxSize - minSize)) + minSize;
                }

                int x = rand() % (t.m().cols - size);
                int y = rand() % (t.m().rows - size);

                if (filterDarkPatches){
                    Mat patch;
                    t.m()(cv::Rect(x,y,size,size)).copyTo(patch);
                    cv::threshold(patch,patch,5,1,cv::THRESH_BINARY);
                    Scalar sumForeground = sum(patch);
                    float bgFgRatio = sumForeground[0] / float(minSize * minSize);
                    if (bgFgRatio < bgFgThresh)
                        continue;
                }

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
