#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/common.h"
#include "openbr/core/qtutils.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

// Because MSVC doesn't provide a round() function in math.h
static int round(float x) { return (floor(x + 0.5)); }

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
    Q_PROPERTY(br::Transform *transform READ get_transform WRITE set_transform RESET reset_transform STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(int stepSize READ get_stepSize WRITE set_stepSize RESET reset_stepSize STORED false)
    Q_PROPERTY(bool takeFirst READ get_takeFirst WRITE set_takeFirst RESET reset_takeFirst STORED false)
    Q_PROPERTY(bool negSamples READ get_negSamples WRITE set_negSamples RESET reset_negSamples STORED false)
    Q_PROPERTY(int negToPosRatio READ get_negToPosRatio WRITE set_negToPosRatio RESET reset_negToPosRatio STORED false)
    Q_PROPERTY(double maxOverlap READ get_maxOverlap WRITE set_maxOverlap RESET reset_maxOverlap STORED false)
    Q_PROPERTY(int windowWidth READ get_windowWidth WRITE set_windowWidth RESET reset_windowWidth STORED false)
    BR_PROPERTY(br::Transform *, transform, NULL)
    BR_PROPERTY(int, minSize, 8)
    BR_PROPERTY(int, stepSize, 1)
    BR_PROPERTY(bool, takeFirst, true)
    BR_PROPERTY(bool, negSamples, true)
    BR_PROPERTY(int, negToPosRatio, 1)
    BR_PROPERTY(double, maxOverlap, 0)
    BR_PROPERTY(int, windowWidth, 24)

public:
    SlidingWindowTransform() : Transform(false, true) {}
private:

    void train(const TemplateList &data)
    {
        // Find avg aspect ratio
        double tempRatio = 0;
        int ratioCnt = 0;
        foreach (const Template &tmpl, data) {
            QList<Rect> posRects = OpenCVUtils::toRects(tmpl.file.rects());
            foreach (const Rect &posRect, posRects) {
                if (posRect.x + posRect.width >= tmpl.m().cols || posRect.y + posRect.height >= tmpl.m().rows || posRect.x < 0 || posRect.y < 0) {
                    continue;
                }
                tempRatio += (float)posRect.width / (float)posRect.height;
                ratioCnt += 1;
            }
        }
        aspectRatio = tempRatio / (double)ratioCnt;

        if (transform->trainable) {
            TemplateList full;

            foreach (const Template &tmpl, data) {
                QList<Rect> posRects = OpenCVUtils::toRects(tmpl.file.rects());
                QList<Rect> negRects;
                foreach (Rect posRect, posRects) {

                    //Adjust for training samples that have different aspect ratios
                    int diff = posRect.width - (int)((float) posRect.height * aspectRatio);
                    posRect.x += diff / 2;
                    posRect.width += diff;

                    if (posRect.x + posRect.width >= tmpl.m().cols || posRect.y + posRect.height >= tmpl.m().rows || posRect.x < 0 || posRect.y < 0) {
                        continue;
                    }

                    Mat scaledImg;
                    resize(Mat(tmpl, posRect), scaledImg, Size(windowWidth,round(windowWidth / aspectRatio)));
                    Template pos(tmpl.file, scaledImg);
                    full += pos;

                    // add random negative samples
                    if (negSamples) {
                        Mat m = tmpl.m();
                        int sample = 0;
                        while (sample < negToPosRatio) {
                            int x = Common::RandSample(1, m.cols)[0];
                            int y = Common::RandSample(1, m.rows)[0];
                            int maxWidth = m.cols - x, maxHeight = m.rows - y;
                            int maxSize = std::min(maxWidth, maxHeight);
                            int size = (maxSize <= minSize ? maxSize : Common::RandSample(1, maxSize, minSize)[0]);
                            Rect negRect(x, y, size, size);
                            // the negative samples cannot overlap the positive at all
                            // but they may overlap with other negatives
                            if (overlaps(posRects, negRect, 0) || overlaps(negRects, negRect, maxOverlap))
                                continue;
                            negRects.append(negRect);
                            Template neg(tmpl.file, Mat(tmpl, negRect));
                            neg.file.set("Label", QString("neg"));
                            full += neg;
                            sample++;
                        }
                    }
                }
            }
            transform->train(full);
        }
    }

    bool overlaps(QList<Rect> posRects, Rect negRect, double overlap)
    {
        foreach (const Rect posRect, posRects) {
            Rect intersect = negRect & posRect;
            if (intersect.area() > overlap*posRect.area())
                return true;
        }
        return false;
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        // no need to slide a window over ground truth data
        if (src.file.getBool("Train", false)) return;

        dst.file.clearRects();
        int windowHeight = (int) round((float) windowWidth / aspectRatio);
        int scale = src.file.get<float>("scale", 1);

        for (double y = 0; y + windowHeight < src.m().rows; y += stepSize) {
            for (double x = 0; x + windowWidth < src.m().cols; x += stepSize) {
                Rect window(x, y, windowWidth, windowHeight);
                Template windowMat(src.file, Mat(src, window));
                Template detect;
                transform->project(windowMat, detect);
                float conf = detect.file.get<float>("conf");

                // the result will be in the Label
                if (conf > 0) {
                    dst.file.appendRect(QRectF((float) x * scale, (float) y * scale, (float) windowWidth * scale, (float) windowHeight * scale));
                    QList<float> confidences = dst.file.getList<float>("Confidences", QList<float>());
                    confidences.append(conf);
                    dst.file.setList<float>("Confidences", confidences);
                    if (takeFirst)
                        return;
                }
            }
        }
    }

    float aspectRatio;
};

BR_REGISTER(Transform, SlidingWindowTransform)

/*!
 * \ingroup transforms
 * \brief .
 * \author Austin Blanton \cite imaus10
 */
class BuildScalesTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform *transform READ get_transform WRITE set_transform RESET reset_transform STORED false)
    Q_PROPERTY(double scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(bool takeLargestScale READ get_takeLargestScale WRITE set_takeLargestScale RESET reset_takeLargestScale STORED false)
    BR_PROPERTY(br::Transform *, transform, NULL)
    BR_PROPERTY(double, scaleFactor, 0.75)
    BR_PROPERTY(bool, takeLargestScale, true)

public:
    BuildScalesTransform() : Transform(false, true) {}
private:

    void train(const TemplateList &data)
    {
        if (transform->trainable)
            transform->train(data);
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        // do not scale images during training
        if (src.file.getBool("Train", false)) return;

        int rows = src.m().rows;
        int cols = src.m().cols;
        for (float scale = 1.0; scale >= 1.0; scale -= (1.0 - scaleFactor)) {
            Template scaleImg(src.file, Mat());
            scaleImg.file.set("scale", scale);
            resize(src, scaleImg, Size(round(cols / scale), round(rows / scale)));
            transform->project(scaleImg, dst);
            if (takeLargestScale && !dst.file.rects().empty())
                return;
        }
    }
};

BR_REGISTER(Transform, BuildScalesTransform)

/*!
 * \ingroup transforms
 * \brief Detects objects with OpenCV's built-in HOG detection.
 * \author Austin Blanton \cite imaus10
 */
class HOGDetectTransform : public UntrainableTransform
{
    Q_OBJECT

    HOGDescriptor hog;

    void init()
    {
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        std::vector<Rect> objLocs;
        QList<Rect> rects;
        hog.detectMultiScale(src, objLocs);
        foreach (const Rect &obj, objLocs)
            rects.append(obj);
        dst.file.setRects(rects);
    }
};

BR_REGISTER(Transform, HOGDetectTransform)

} // namespace br

#include "slidingwindow.moc"
