#include <opencv2/features2d/features2d.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV Key Point Detector
 * \author Josh Klontz \cite jklontz
 */
class KeyPointDetectorTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString detector READ get_detector WRITE set_detector RESET reset_detector STORED false)
    BR_PROPERTY(QString, detector, "SIFT")

    Ptr<FeatureDetector> featureDetector;

    void init()
    {
        featureDetector = FeatureDetector::create(detector.toStdString());
        if (featureDetector.empty())
            qFatal("Failed to create KeyPointDetector: %s", qPrintable(detector));
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        std::vector<KeyPoint> keyPoints;
        try {
            featureDetector->detect(src, keyPoints);
        } catch (...) {
            qWarning("Key point detection failed for file %s", qPrintable(src.file.name));
            dst.file.fte = true;
        }

        QList<Rect> rects;
        foreach (const KeyPoint &keyPoint, keyPoints)
            rects.append(Rect(keyPoint.pt.x-keyPoint.size/2, keyPoint.pt.y-keyPoint.size/2, keyPoint.size, keyPoint.size));
        dst.file.setRects(OpenCVUtils::fromRects(rects));
    }
};

BR_REGISTER(Transform, KeyPointDetectorTransform)

} // namespace br

#include "keypointdetector.moc"
