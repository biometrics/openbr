#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Applies a transform to a sliding window.
 *        Discards negative detections.
 * \author Austin Blanton \cite imaus10
 */
class SlidingWindowTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(double scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(double stepSize READ get_stepSize WRITE set_stepSize RESET reset_stepSize STORED false)
    Q_PROPERTY(bool takeLargestScale READ get_takeLargestScale WRITE set_takeLargestScale RESET reset_takeLargestScale STORED false)
    Q_PROPERTY(br::Transform *transform READ get_transform WRITE set_transform RESET reset_transform STORED false)
    BR_PROPERTY(int, minSize, 8)
    BR_PROPERTY(double, scaleFactor, 0.75)
    BR_PROPERTY(double, stepSize, 1)
    BR_PROPERTY(bool, takeLargestScale, true)
    BR_PROPERTY(br::Transform *, transform, NULL)

public:
    SlidingWindowTransform() : Transform(false, true) {}

private:
    void train(const TemplateList &data)
    {
        if (transform->trainable) {
            TemplateList full;
            foreach (const Template &tmpl, data) {
                foreach (const Rect &rect, OpenCVUtils::toRects(tmpl.file.rects())) {
                    Template pos(tmpl.file, Mat(tmpl, rect));
                    // add the positive label
                    pos.file.set("Label", 1);
                    full += pos;

                    // add negative overlapping samples at the same scale
                    // false positives are most likely overlapping
                    // TODO: parameterize negToPosRatio
                    int negToPosRatio = 9;
                    for (int sample=0; sample<negToPosRatio; sample++) {
                        int x, y;
                        if (sample/3 == 0)
                            y = rect.y - rect.height/2;
                        else if (sample/3 == 1)
                            y = rect.y;
                        else
                            y = rect.y + rect.height/2;

                        if (sample%3 == 0)
                            x = rect.x - rect.width/2;
                        else if (sample%3 == 1)
                            x = rect.x;
                        else
                            x = rect.x + rect.width/2;

                        if (x < 0 || y < 0 || x + rect.width >= tmpl.m().cols || y + rect.height >= tmpl.m().rows)
                            continue;
                        Template neg(tmpl.file, Mat(tmpl, Rect(x, y, rect.width, rect.height)));
                        neg.file.set("Label", QVariant::fromValue(0));
                        full += neg;
                    }
                }
            }
            transform->train(full);
        }
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        int rows = src.m().rows, cols = src.m().cols;
        for (int size=std::min(rows, cols); size>minSize; size*=scaleFactor) {
            for (int y=0; y+size<rows; y+=stepSize) {
                for (int x=0; x+size>cols; x+=stepSize) {
                    Template window(src.file, Mat(src.m(), Rect(x, y, size, size)));
                    Template detect;
                    transform->projectUpdate(window, detect);
                    // the result should be a single binary classification for the window:
                    // detection (1) or not (0)
                    if (countNonZero(detect) != 0) {
                        dst += window;
                        if (takeLargestScale) return;
                    }
                }
            }
        }
    }

    void project(const Template &src, Template &dst) const
    {
        (void) src; (void) dst; qFatal("what happened");
    }
};

BR_REGISTER(Transform, SlidingWindowTransform)

} // namespace br

#include "slidingwindow.moc"
