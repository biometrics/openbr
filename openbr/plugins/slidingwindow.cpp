#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/common.h"

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
    Q_PROPERTY(bool negSamples READ get_negSamples WRITE set_negSamples RESET reset_negSamples STORED false)
    Q_PROPERTY(int negToPosRatio READ get_negToPosRatio WRITE set_negToPosRatio RESET reset_negToPosRatio STORED false)
    BR_PROPERTY(int, minSize, 8)
    BR_PROPERTY(double, scaleFactor, 0.75)
    BR_PROPERTY(double, stepSize, 1)
    BR_PROPERTY(bool, takeLargestScale, true)
    BR_PROPERTY(br::Transform *, transform, NULL)
    BR_PROPERTY(bool, negSamples, true)
    BR_PROPERTY(int, negToPosRatio, 1)

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

                    // add random negative samples
                    if (negSamples) {
                        Mat m = tmpl.m();
                        QList<int> xs = Common::RandSample(negToPosRatio, m.cols, 0, true);
                        QList<int> ys = Common::RandSample(negToPosRatio, m.rows, 0, true);
                        for (int i=0; i<negToPosRatio; i++) {
                            int x = xs[i], y = ys[i];
                            int maxWidth = m.cols - x, maxHeight = m.rows - y;
                            int maxSize = std::min(maxWidth, maxHeight);
                            int size = (maxSize < minSize ? maxSize : Common::RandSample(1, maxSize, minSize)[0]);
                            Template neg(tmpl.file, Mat(tmpl, Rect(x, y, size, size)));
                            neg.file.set("Label", QVariant::fromValue(0));
                            full += neg;
                        }
                    }
                }
            }
            transform->train(full);
        }
    }

    void project(const Template &src, Template &dst) const
    {
        int rows = src.m().rows, cols = src.m().cols;
        for (int size=std::min(rows, cols); size>=minSize; size*=scaleFactor) {
            for (int y=0; y+size<rows; y+=stepSize) {
                for (int x=0; x+size>cols; x+=stepSize) {
                    Template window(src.file, Mat(src.m(), Rect(x, y, size, size)));
                    Template detect;
                    transform->project(window, detect);
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
};

BR_REGISTER(Transform, SlidingWindowTransform)

} // namespace br

#include "slidingwindow.moc"
