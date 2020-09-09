#include "openbr/core/evalutils.h"
#include "openbr/core/qtutils.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/common.h"

#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace br;
using namespace cv;
using namespace EvalUtils;

static const int Max_Points = 500; // Maximum number of points to render on plots

DetectionKey EvalUtils::getDetectKey(const FileList &files)
{
    if (files.empty())
        return DetectionKey();

    const File &f = files.first();
    const QStringList localKeys = f.localKeys();

    // first check for single detections
    foreach (const QString &key, localKeys)
        if (!f.get<QRectF>(key, QRectF()).isNull())
            return DetectionKey(key, DetectionKey::Rect);

    // and then multiple
    if (!f.rects().empty())
        return DetectionKey("Rects", DetectionKey::RectList);

    // check for <Key>_X, <Key>_Y, <Key>_Width, <Key>_Height
    foreach (const QString &localKey, localKeys) {
        if (!localKey.endsWith("_X"))
            continue;
        const QString key = localKey.mid(0, localKey.size()-2);
        if (localKeys.contains(key+"_Y") &&
            localKeys.contains(key+"_Width") &&
            localKeys.contains(key+"_Height"))
            return DetectionKey(key, DetectionKey::XYWidthHeight);
    }

    return DetectionKey();
}

// return a list of detections independent of the detection key format
QList<Detection> EvalUtils::getDetections(const DetectionKey &key, const File &f, bool isTruth)
{
    QString pose = f.get<QString>("Pose");
    if (pose.contains("Angle"))
        pose = "Frontal";

    const QString filePath = f.path() + "/" + f.fileName();
    QList<Detection> dets;
    if (key.type == DetectionKey::RectList) {
        QList<QRectF> rects = f.rects();
        QList<float> confidences = f.getList<float>("Confidences", QList<float>());
        if (!isTruth && rects.size() != confidences.size())
            qFatal("You don't have enough confidence. I mean, your detections don't all have confidence measures.");
        for (int i=0; i<rects.size(); i++) {
            if (isTruth)
                dets.append(Detection(rects[i], filePath));
            else
                dets.append(Detection(rects[i], filePath, confidences[i]));
        }
    } else if (key.type == DetectionKey::Rect) {
        dets.append(Detection(f.get<QRectF>(key), filePath, isTruth ? -1 : f.get<float>("Confidence", -1), f.get<bool>("Ignore", false), pose));
    } else if (key.type == DetectionKey::XYWidthHeight) {
        const QRectF rect(f.get<float>(key+"_X"), f.get<float>(key+"_Y"), f.get<float>(key+"_Width"), f.get<float>(key+"_Height"));
        dets.append(Detection(rect, filePath, isTruth ? -1 : f.get<float>("Confidence", -1), f.get<bool>("Ignore", false), pose));
    }
    return dets;
}

QMap<QString, Detections> EvalUtils::getDetections(const File &predictedGallery, const File &truthGallery)
{
    const FileList predicted = TemplateList::fromGallery(predictedGallery).files();
    const FileList truth = TemplateList::fromGallery(truthGallery).files();

    // Figure out which metadata field contains a bounding box
    DetectionKey truthDetectKey = getDetectKey(truth);
    if (truthDetectKey.isEmpty())
        qFatal("No suitable ground truth metadata key found.");

    DetectionKey predictedDetectKey = getDetectKey(predicted);
    if (predictedDetectKey.isEmpty())
        qFatal("No suitable predicted metadata key found.");

    if (Globals->verbose) {
        qDebug("Using metadata key: %s%s",
               qPrintable(predictedDetectKey),
               qPrintable(predictedDetectKey == truthDetectKey ? QString() : "/"+truthDetectKey));
    }

    QMap<QString, Detections> allDetections;
    foreach (const File &f, truth) {
        allDetections[f.name].truth.append(getDetections(truthDetectKey, f, true));
        const cv::Mat image = cv::imread(qPrintable(br::Globals->path + QDir::separator() + f.name));
        allDetections[f.name].imageSize = QSize(image.cols, image.rows);
    }
    foreach (const File &f, predicted)
        if (allDetections.contains(f.name))
            allDetections[f.name].predicted.append(getDetections(predictedDetectKey, f, false));
    return allDetections;
}

QMap<QString, Detections> EvalUtils::filterDetections(const QMap<QString, Detections> &allDetections, int threshold, bool useMin, float relativeThreshold)
{
    QMap<QString, Detections> allFilteredDetections;
    foreach (QString key, allDetections.keys()) {
        Detections detections = allDetections[key];
        const int adaptiveThreshold = qMax((int)(qMax(detections.imageSize.width(), detections.imageSize.height()) * relativeThreshold + 0.5f), threshold);
        Detections filteredDetections;
        for (int i = 0; i < detections.predicted.size(); i++) {
            const QRectF box = detections.predicted[i].boundingBox;
            const qreal maxBoxDim = max(box.width(), box.height());
            const double t = sqrt(0.5 * adaptiveThreshold * adaptiveThreshold);
            if (useMin ? maxBoxDim > t : maxBoxDim < t)
                filteredDetections.predicted.append(detections.predicted[i]);
        }
        for (int i = 0; i < detections.truth.size(); i++) {
            const QRectF box = detections.truth[i].boundingBox;
            const qreal maxBoxDim = max(box.width(), box.height());
            if (useMin ? maxBoxDim < threshold : maxBoxDim > threshold)
                detections.truth[i].ignore = true;
            filteredDetections.truth.append(detections.truth[i]);
        }
        if (!filteredDetections.truth.empty()) allFilteredDetections[key] = filteredDetections;
    }
    return allFilteredDetections;
}

int EvalUtils::associateGroundTruthDetections(QList<ResolvedDetection> &resolved, QList<ResolvedDetection> &falseNegative, QMap<QString, Detections> &all, QRectF &offsets)
{
    QList<float> dLeft, dRight, dTop, dBottom;
    int count = 0, totalTrueDetections = 0;

    foreach (Detections detections, all.values()) {
        for (int i=0; i<detections.truth.size(); i++)
            if (!detections.truth[i].ignore) totalTrueDetections++;

        // Try to associate ground truth detections with predicted detections
        QList<SortedDetection> sortedDetections; sortedDetections.reserve(detections.truth.size() * detections.predicted.size());
        for (int t = 0; t < detections.truth.size(); t++) {
            const Detection truth = detections.truth[t];
            for (int p = 0; p < detections.predicted.size(); p++) {
                Detection predicted = detections.predicted[p];

                float predictedWidth = predicted.boundingBox.width();
                float x, y, width, height;
                x = predicted.boundingBox.x() + offsets.x()*predictedWidth;
                y = predicted.boundingBox.y() + offsets.y()*predictedWidth;
                width = predicted.boundingBox.width() - offsets.width()*predictedWidth;
                height = predicted.boundingBox.height() - offsets.height()*predictedWidth;
                Detection newPredicted(QRectF(x, y, width, height), predicted.filePath, 0.0);

                const float overlap = truth.overlap(newPredicted);

                if (overlap > 0)
                    sortedDetections.append(SortedDetection(t, p, overlap));
            }
        }

        std::sort(sortedDetections.begin(), sortedDetections.end());

        QList<int> removedTruth;
        QList<int> removedPredicted;

        foreach (const SortedDetection &detection, sortedDetections) {
            if (removedTruth.contains(detection.truth_idx) || removedPredicted.contains(detection.predicted_idx))
                continue;

            const Detection truth = detections.truth[detection.truth_idx];
            const Detection predicted = detections.predicted[detection.predicted_idx];

            if (!truth.ignore)
                resolved.append(ResolvedDetection(predicted.filePath, predicted.boundingBox, predicted.confidence, detection.overlap, truth.boundingBox, truth.pose == predicted.pose));

            removedTruth.append(detection.truth_idx);
            removedPredicted.append(detection.predicted_idx);

            if (offsets.x() == 0 && detection.overlap > 0.3) {
                count++;
                float width = predicted.boundingBox.width();
                dLeft.append((truth.boundingBox.left() - predicted.boundingBox.left()) / width);
                dRight.append((truth.boundingBox.right() - predicted.boundingBox.right()) / width);
                dTop.append((truth.boundingBox.top() - predicted.boundingBox.top()) / width);
                dBottom.append((truth.boundingBox.bottom() - predicted.boundingBox.bottom()) / width);
            }
        }

        // False positive
        for (int i = 0; i < detections.predicted.size(); i++)
            if (!removedPredicted.contains(i)) resolved.append(ResolvedDetection(detections.predicted[i].filePath, detections.predicted[i].boundingBox, detections.predicted[i].confidence, 0, QRectF(), false));

        // False negative
        for (int i = 0; i < detections.truth.size(); i++)
            if (!removedTruth.contains(i) && !detections.truth[i].ignore) falseNegative.append(ResolvedDetection(detections.truth[i].filePath, detections.truth[i].boundingBox, -std::numeric_limits<float>::max(), 0, QRectF(), false));
    }

    if (offsets.x() == 0) {
        // Calculate average differences in each direction
        double dTopMean, dTopStdDev;
        Common::MeanStdDev(dTop,&dTopMean,&dTopStdDev);
        double dBottomMean, dBottomStdDev;
        Common::MeanStdDev(dBottom,&dBottomMean,&dBottomStdDev);
        double dLeftMean, dLeftStdDev;
        Common::MeanStdDev(dLeft,&dLeftMean,&dLeftStdDev);
        double dRightMean, dRightStdDev;
        Common::MeanStdDev(dRight,&dRightMean,&dRightStdDev);
        double dWidth = dLeftMean - dRightMean;
        double dHeight = dTopMean - dBottomMean;

        offsets.setX(dLeftMean);
        offsets.setY(dTopMean);
        offsets.setWidth(dWidth);
        offsets.setHeight(dHeight);

        if (Globals->verbose) {
            qDebug("oLeft = %.3f", dLeftStdDev);
            qDebug("oRight = %.3f", dRightStdDev);
            qDebug("oTop = %.3f", dTopStdDev);
            qDebug("oBottom = %.3f", dBottomStdDev);
        }
    }
    return totalTrueDetections;
}

QStringList EvalUtils::computeDetectionResults(const QList<ResolvedDetection> &detections, int totalTrueDetections, int numImages, bool discrete, QList<DetectionOperatingPoint> &points)
{
    float TP = 0, FP = 0, prevFP = -1, prevTP = -1;

    QList<float> FARsToOutput;
    FARsToOutput << 1 << .5 << .2 << .1 << .02 << .01 << .001;

    QDebug debug = qDebug();
    debug.noquote();

    debug << endl << QString("+")+QString("-").repeated(12)+QString("+")+QString("-").repeated(12)+QString("+")+QString("-").repeated(12)+QString("+")+QString("-").repeated(12)+QString("+") << endl;
    debug << QString("|") << QString("FAR").leftJustified(10, ' ') << QString("|") << QString("TAR").leftJustified(10, ' ') << QString("|") << QString("Confidence").leftJustified(10, ' ') << QString("|") << QString("Pose Match").leftJustified(10, ' ') << QString("|") << endl;
    debug << QString("+")+QString("-").repeated(12)+QString("+")+QString("-").repeated(12)+QString("+")+QString("-").repeated(12)+QString("+")+QString("-").repeated(12)+QString("+") << endl;

    float poseMatch = 0;
    int detectionsToKeep = 50;
    QList<ResolvedDetection> topFalsePositives, bottomTruePositives;
    for (int i=0; i<detections.size(); i++) {
        const ResolvedDetection &detection = detections[i];
        if (discrete) {
            // A 50% overlap is considered a true positive
            if (detection.overlap >= 0.5) {
                TP++;
                if (detection.poseMatch)
                    poseMatch++;
            }
            else
                FP++;
        } else {
            TP += detection.overlap;
            if (detection.poseMatch)
                poseMatch += detection.overlap;
            FP += 1 - detection.overlap;
        }
        if ((i == detections.size()-1) || (detection.confidence > detections[i+1].confidence)) {
            if (FP > prevFP || (i == detections.size()-1)) {
                foreach (float FAR, FARsToOutput)
                    if (prevFP / numImages < FAR && FP / numImages >= FAR) {
                        debug << QString("|") << QString::number(FAR, 'f', 4).leftJustified(10, ' ');
                        debug << QString("|") << QString::number(totalTrueDetections ? TP / totalTrueDetections : 0, 'f', 4).leftJustified(10, ' ');
                        debug << QString("|") << QString::number(detection.confidence, 'f', 4).leftJustified(10, ' ');
                        debug << QString("|") << QString::number(TP ? poseMatch / TP : 0., 'f', 4).leftJustified(10, ' ');
                        debug << QString("|") << endl;
                        break;
                    }

                if (detection.overlap < 0.5 && topFalsePositives.size() < detectionsToKeep)
                    topFalsePositives.append(detection);

                points.append(DetectionOperatingPoint(TP, FP, totalTrueDetections, numImages, detection.confidence));
                prevFP = FP;
            }

            if (TP > prevTP) {
                bottomTruePositives.append(detection);
                if (bottomTruePositives.size() > detectionsToKeep)
                    bottomTruePositives.removeFirst();
                prevTP = TP;
            }
        }
    }

    debug << QString("+")+QString("-").repeated(12)+QString("+")+QString("-").repeated(12)+QString("+")+QString("-").repeated(12)+QString("+")+QString("-").repeated(12)+QString("+") << endl;

    if (discrete) {
        if (Globals->verbose) {
            QtUtils::touchDir(QDir("./falsePos"));
            qDebug("Highest Scoring False Positives:");
	    detectionsToKeep = std::min(detectionsToKeep,topFalsePositives.size());
            for (int i=0; i<detectionsToKeep; i++) {
                Mat img = imread(qPrintable(Globals->path + "/" + topFalsePositives[i].filePath));
                const Rect falseRect = OpenCVUtils::toRect(topFalsePositives[i].boundingBox);
                rectangle(img, falseRect, Scalar(0, 0, 255), 1);
                rectangle(img, OpenCVUtils::toRect(topFalsePositives[i].groundTruthBoundingBox), Scalar(0, 255, 0), 1);
                putText(img, qPrintable("Overlap:"+QString::number(topFalsePositives[i].overlap)), falseRect.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
                imwrite(qPrintable(QString("./falsePos/falsePos%1.jpg").arg(QString::number(i))), img);
            }
            qDebug("Lowest Scoring True Positives:");
            qDebug() << bottomTruePositives;
        }
    }

    debug << QString("Minimum %1 Precision:").arg(discrete ? "Discrete" : "Continuous").leftJustified(32, ' ') << QString("%1").arg(QString::number(points.last().Precision)) << endl;
    debug << QString("Maximum %1 Recall:").arg(discrete ? "Discrete" : "Continuous").leftJustified(32, ' ') << QString("%1").arg(QString::number(points.last().Recall)) << endl;
    debug << QString("%1 F1 Score:").arg(discrete ? "Discrete" : "Continuous").leftJustified(32, ' ') << QString("%1").arg(QString::number((points.last().Recall + points.last().Precision) ? 2 * (points.last().Recall * points.last().Precision) / (points.last().Recall + points.last().Precision) : 0)) << endl;

    const int keep = qMin(points.size(), Max_Points);
    if (keep < 1) qFatal("Insufficient points.");

    QStringList lines; lines.reserve(keep);
    if (keep == 1) {
        const DetectionOperatingPoint &point = points[0];
        lines.append(QString("%1ROC, %2, %3").arg(discrete ? "Discrete" : "Continuous", QString::number(point.FalsePositiveRate), QString::number(point.Recall)));
        lines.append(QString("%1PR, %2, %3").arg(discrete ? "Discrete" : "Continuous", QString::number(point.Recall), QString::number(point.Precision)));
    } else {
        for (int i=0; i<keep; i++) {
            const DetectionOperatingPoint &point = points[double(i) / double(keep-1) * double(points.size()-1)];
            lines.append(QString("%1ROC, %2, %3").arg(discrete ? "Discrete" : "Continuous", QString::number(point.FalsePositiveRate), QString::number(point.Recall)));
            lines.append(QString("%1PR, %2, %3").arg(discrete ? "Discrete" : "Continuous", QString::number(point.Recall), QString::number(point.Precision)));
        }
    }
    return lines;
}

QDebug operator<<(QDebug dbg, const ResolvedDetection &d)
{
    return dbg.nospace() << "(FilePath: " << d.filePath << " Bounding Box: " << d.boundingBox << ", Overlap: " << d.overlap << ", Confidence: " << d.confidence << ")";
}
