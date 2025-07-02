#ifndef EVALUTILS_EVALUTILS_H
#define EVALUTILS_EVALUTILS_H

#include <openbr/openbr_plugin.h>
#include <openbr/core/qtutils.h>
#include <opencv2/highgui/highgui.hpp>

namespace EvalUtils
{

struct Detection
{
    QRectF boundingBox;
    QString filePath;
    float confidence;
    // The ignore flag is useful when certain faces in an image should be ignored
    // and should not effect detection performance. Predicted detections that overlap
    // with an ignored truth detection will not count as a true positive, false positive,
    // true negative, or false negative, it will simply be ignored.
    bool ignore;
    QString pose;
    // The label field can be used to distinguish between different object classes
    QString label;

    Detection() {}
    Detection(const QRectF &boundingBox, const QString &filePath = QString(), float confidence = -1, bool ignore = false, const QString &pose = "Frontal", const QString &label = "") :
        boundingBox(boundingBox),
        filePath(filePath),
        confidence(confidence),
        ignore(ignore),
        pose(pose),
        label(label)
    {}

    float overlap(const Detection &other) const
    {
        return QtUtils::overlap(boundingBox, other.boundingBox);
    }
};

struct SortedDetection
{
    int truth_idx, predicted_idx;
    float overlap, confidence, truePositiveThreshold;
    SortedDetection() : truth_idx(-1), predicted_idx(-1), overlap(-1), confidence(-1), truePositiveThreshold(-1) {}
    SortedDetection(int truth_idx_, int predicted_idx_, float overlap_, float confidence_, float truePositiveThreshold_)
        : truth_idx(truth_idx_), predicted_idx(predicted_idx_), overlap(overlap_), confidence(confidence_), truePositiveThreshold(truePositiveThreshold_) {}
    inline bool operator<(const SortedDetection &other) const 
    { 
        if (overlap >= truePositiveThreshold && other.overlap >= truePositiveThreshold)
            return confidence > other.confidence;
        return overlap > other.overlap; 
    }
};

struct ResolvedDetection
{
    QString filePath;
    QRectF boundingBox, groundTruthBoundingBox;
    float confidence, overlap;
    bool poseMatch;
    QString label;
    ResolvedDetection() :
    confidence(-1),
        overlap(-1)
        {}

ResolvedDetection(const QString &filePath, const QRectF &boundingBox, float confidence, float overlap, const QRectF &groundTruthBoundingBox, bool poseMatch, const QString &label) :
    filePath(filePath),
    boundingBox(boundingBox),
    groundTruthBoundingBox(groundTruthBoundingBox),
    confidence(confidence),
    overlap(overlap),
    poseMatch(poseMatch),
    label(label)
    {}

    inline bool operator<(const ResolvedDetection &other) const { return confidence > other.confidence; }
};

struct Detections
{
    QList<Detection> predicted, truth;
    QSize imageSize;
};

struct DetectionKey : public QString
{
    enum Type {
        Invalid,
        Rect,
        RectList,
        XYWidthHeight
    } type;

    DetectionKey(const QString &key = "", Type type = Invalid)
        : QString(key), type(type) {}
};

struct DetectionOperatingPoint
{
    float Recall, FalsePositiveRate, Precision, Confidence;
    DetectionOperatingPoint() : Recall(-1), FalsePositiveRate(-1), Precision(-1) {}
    DetectionOperatingPoint(float TP, float FP, float totalPositives, float numImages, float confidence)
        : Recall(totalPositives ? TP/totalPositives : 0), FalsePositiveRate(FP/numImages), Precision(TP/(TP+FP)), Confidence(confidence) {}
};

    // Detection
    DetectionKey getDetectKey(const br::FileList &files);
    QList<Detection> getDetections(const DetectionKey &key, const br::File &f, bool isTruth);
    QMap<QString, Detections> getDetections(const br::File &predictedGallery, const br::File &truthGallery);
    QMap<QString, Detections> filterDetections(const QMap<QString, Detections> &allDetections, int threshold, bool useMin = true, float relativeThreshold = 0);
    QMap<QString, Detections> filterLabels(const QMap<QString, Detections> &allDetections, const QString &label);
    int associateGroundTruthDetections(QList<ResolvedDetection> &resolved, QList<ResolvedDetection> &falseNegative, QMap<QString, Detections> &all, QRectF &offsets, float truePositiveThreshold);
    QStringList computeDetectionResults(const QList<ResolvedDetection> &detections, int totalTrueDetections, int numImages, bool discrete, QList<DetectionOperatingPoint> &points, const float truePositiveThreshold);
    inline int getNumberOfImages(const QMap<QString, Detections> detections)
    {
        return detections.keys().size();
    }
}

QDebug operator<<(QDebug dbg, const EvalUtils::ResolvedDetection &d);

#endif // EVALUTILS_EVALUTILS_H
