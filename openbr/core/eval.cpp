/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "bee.h"
#include "eval.h"
#include "openbr/core/common.h"
#include "openbr/core/qtutils.h"

using namespace cv;

namespace br
{

static const int Max_Points = 500; // Maximum number of points to render on plots

struct Comparison
{
    float score;
    int target, query;
    bool genuine;

    Comparison() {}
    Comparison(float _score, int _target, int _query, bool _genuine)
        : score(_score), target(_target), query(_query), genuine(_genuine) {}
    inline bool operator<(const Comparison &other) const { return score > other.score; }
};

#undef FAR // Windows preprecessor definition conflicts with variable name
struct OperatingPoint
{
    float score, FAR, TAR;
    OperatingPoint() {}
    OperatingPoint(float _score, float _FAR, float _TAR)
        : score(_score), FAR(_FAR), TAR(_TAR) {}
};

static float getTAR(const QList<OperatingPoint> &operatingPoints, float FAR)
{
    int index = 0;
    while (operatingPoints[index].FAR < FAR) {
        index++;
        if (index == operatingPoints.size())
            return 1;
    }

    const float x1 = (index == 0 ? 0 : operatingPoints[index-1].FAR);
    const float y1 = (index == 0 ? 0 : operatingPoints[index-1].TAR);
    const float x2 = operatingPoints[index].FAR;
    const float y2 = operatingPoints[index].TAR;
    const float m = (y2 - y1) / (x2 - x1);
    const float b = y1 - m*x1;
    return m * FAR + b;
}

float Evaluate(const cv::Mat &scores, const FileList &target, const FileList &query, const QString &csv, int partition)
{
    return Evaluate(scores, BEE::makeMask(target, query, partition), csv);
}

float Evaluate(const QString &simmat, const QString &mask, const QString &csv)
{
    qDebug("Evaluating %s%s%s",
           qPrintable(simmat),
           mask.isEmpty() ? "" : qPrintable(" with " + mask),
           csv.isEmpty() ? "" : qPrintable(" to " + csv));

    // Read similarity matrix
    QString target, query;
    Mat scores;
    if (simmat.endsWith(".mtx")) {
        scores = BEE::readSimmat(simmat, &target, &query);
    } else {
        QScopedPointer<Format> format(Factory<Format>::make(simmat));
        scores = format->read();
    }

    // Read mask matrix
    Mat truth;
    if (mask.isEmpty()) {
        // Use the galleries specified in the similarity matrix
        if (target.isEmpty()) qFatal("Unspecified target gallery.");
        if (query.isEmpty()) qFatal("Unspecified query gallery.");
        truth = BEE::makeMask(TemplateList::fromGallery(target).files(),
                              TemplateList::fromGallery(query).files());
    } else {
        File maskFile(mask);
        maskFile.set("rows", scores.rows);
        maskFile.set("columns", scores.cols);
        QScopedPointer<Format> format(Factory<Format>::make(maskFile));
        truth = format->read();
    }

    return Evaluate(scores, truth, csv);
}

float Evaluate(const Mat &simmat, const Mat &mask, const QString &csv)
{
    if (simmat.size() != mask.size())
        qFatal("Similarity matrix (%ix%i) differs in size from mask matrix (%ix%i).",
               simmat.rows, simmat.cols, mask.rows, mask.cols);

    float result = -1;

    // Make comparisons
    QList<Comparison> comparisons; comparisons.reserve(simmat.rows*simmat.cols);
    int genuineCount = 0, impostorCount = 0, numNaNs = 0;
    for (int i=0; i<simmat.rows; i++) {
        for (int j=0; j<simmat.cols; j++) {
            const BEE::Mask_t mask_val = mask.at<BEE::Mask_t>(i,j);
            const BEE::Simmat_t simmat_val = simmat.at<BEE::Simmat_t>(i,j);
            if (mask_val == BEE::DontCare) continue;
            if (simmat_val != simmat_val) { numNaNs++; continue; }
            comparisons.append(Comparison(simmat_val, j, i, mask_val == BEE::Match));
            if (comparisons.last().genuine) genuineCount++;
            else                            impostorCount++;
        }
    }

    if (numNaNs > 0) qWarning("Encountered %d NaN scores!", numNaNs);
    if (genuineCount == 0) qFatal("No genuine scores!");
    if (impostorCount == 0) qFatal("No impostor scores!");

    // Sort comparisons by simmat_val (score)
    std::sort(comparisons.begin(), comparisons.end());

    QList<OperatingPoint> operatingPoints;
    QList<float> genuines; genuines.reserve(sqrt((float)comparisons.size()));
    QList<float> impostors; impostors.reserve(comparisons.size());
    QVector<int> firstGenuineReturns(simmat.rows, 0);

    int falsePositives = 0, previousFalsePositives = 0;
    int truePositives = 0, previousTruePositives = 0;
    int index = 0;
    float minGenuineScore = std::numeric_limits<float>::max();
    float minImpostorScore = std::numeric_limits<float>::max();

    while (index < comparisons.size()) {
        float thresh = comparisons[index].score;
        // Compute genuine and imposter statistics at a threshold
        while ((index < comparisons.size()) &&
               (comparisons[index].score == thresh)) {
            const Comparison &comparison = comparisons[index];
            if (comparison.genuine) {
                truePositives++;
                genuines.append(comparison.score);
                if (firstGenuineReturns[comparison.query] < 1)
                    firstGenuineReturns[comparison.query] = abs(firstGenuineReturns[comparison.query]) + 1;
                if ((comparison.score != -std::numeric_limits<float>::max()) &&
                    (comparison.score < minGenuineScore))
                    minGenuineScore = comparison.score;
            } else {
                falsePositives++;
                impostors.append(comparison.score);
                if (firstGenuineReturns[comparison.query] < 1)
                    firstGenuineReturns[comparison.query]--;
                if ((comparison.score != -std::numeric_limits<float>::max()) &&
                    (comparison.score < minImpostorScore))
                    minImpostorScore = comparison.score;
            }
            index++;
        }

        if ((falsePositives > previousFalsePositives) &&
             (truePositives > previousTruePositives)) {
            operatingPoints.append(OperatingPoint(thresh, float(falsePositives)/impostorCount, float(truePositives)/genuineCount));
            previousFalsePositives = falsePositives;
            previousTruePositives = truePositives;
        }
    }

    if (operatingPoints.size() == 0) operatingPoints.append(OperatingPoint(1, 1, 1));
    if (operatingPoints.size() == 1) operatingPoints.prepend(OperatingPoint(0, 0, 0));
    if (operatingPoints.size() > 2)  operatingPoints.takeLast(); // Remove point (1,1)

    // Write Metadata table
    QStringList lines;
    lines.append("Plot,X,Y");
    lines.append("Metadata,"+QString::number(simmat.cols)+",Gallery");
    lines.append("Metadata,"+QString::number(simmat.rows)+",Probe");
    lines.append("Metadata,"+QString::number(genuineCount)+",Genuine");
    lines.append("Metadata,"+QString::number(impostorCount)+",Impostor");
    lines.append("Metadata,"+QString::number(simmat.cols*simmat.rows-(genuineCount+impostorCount))+",Ignored");

    // Write Detection Error Tradeoff (DET), PRE, REC
    int points = qMin(operatingPoints.size(), Max_Points);
    for (int i=0; i<points; i++) {
        const OperatingPoint &operatingPoint = operatingPoints[double(i) / double(points-1) * double(operatingPoints.size()-1)];
        lines.append(QString("DET,%1,%2").arg(QString::number(operatingPoint.FAR),
                                              QString::number(1-operatingPoint.TAR)));
        lines.append(QString("FAR,%1,%2").arg(QString::number(operatingPoint.score),
                                              QString::number(operatingPoint.FAR)));
        lines.append(QString("FRR,%1,%2").arg(QString::number(operatingPoint.score),
                                              QString::number(1-operatingPoint.TAR)));
    }

    // Write FAR/TAR Bar Chart (BC)
    lines.append(qPrintable(QString("BC,0.001,%1").arg(QString::number(getTAR(operatingPoints, 0.001), 'f', 3))));
    lines.append(qPrintable(QString("BC,0.01,%1").arg(QString::number(result = getTAR(operatingPoints, 0.01), 'f', 3))));

    // Write SD & KDE
    points = qMin(qMin(Max_Points, genuines.size()), impostors.size());
    QList<double> sampledGenuineScores; sampledGenuineScores.reserve(points);
    QList<double> sampledImpostorScores; sampledImpostorScores.reserve(points);

    if (points > 1) {
        for (int i=0; i<points; i++) {
            float genuineScore = genuines[double(i) / double(points-1) * double(genuines.size()-1)];
            float impostorScore = impostors[double(i) / double(points-1) * double(impostors.size()-1)];
            if (genuineScore == -std::numeric_limits<float>::max()) genuineScore = minGenuineScore;
            if (impostorScore == -std::numeric_limits<float>::max()) impostorScore = minImpostorScore;
            lines.append(QString("SD,%1,Genuine").arg(QString::number(genuineScore)));
            lines.append(QString("SD,%1,Impostor").arg(QString::number(impostorScore)));
            sampledGenuineScores.append(genuineScore);
            sampledImpostorScores.append(impostorScore);
        }
    }

    // Write Cumulative Match Characteristic (CMC) curve
    const int Max_Retrieval = 200;
    const int Report_Retrieval = 5;

    float reportRetrievalRate = -1;
    for (int i=1; i<=Max_Retrieval; i++) {
        int realizedReturns = 0, possibleReturns = 0;
        foreach (int firstGenuineReturn, firstGenuineReturns) {
            if (firstGenuineReturn > 0) {
                possibleReturns++;
                if (firstGenuineReturn <= i) realizedReturns++;
            }
        }
        const float retrievalRate = float(realizedReturns)/possibleReturns;
        lines.append(qPrintable(QString("CMC,%1,%2").arg(QString::number(i), QString::number(retrievalRate))));
        if (i == Report_Retrieval) reportRetrievalRate = retrievalRate;
    }

    QtUtils::writeFile(csv, lines);
    qDebug("TAR @ FAR = 0.01: %.3f\nRetrieval Rate @ Rank = %d: %.3f", result, Report_Retrieval, reportRetrievalRate);
    return result;
}

// Helper struct for statistics accumulation
struct Counter
{
    float truePositive, falsePositive, falseNegative;
    Counter()
    {
        truePositive = 0;
        falsePositive = 0;
        falseNegative = 0;
    }
};

void EvalClassification(const QString &predictedGallery, const QString &truthGallery, QString predictedProperty, QString truthProperty)
{
    qDebug("Evaluating classification of %s against %s", qPrintable(predictedGallery), qPrintable(truthGallery));

    if (predictedProperty.isEmpty())
        predictedProperty = "Label";
    // If predictedProperty is specified, but truthProperty isn't, copy over the value from
    // predicted property
    else if (truthProperty.isEmpty())
        truthProperty = predictedProperty;

    if (truthProperty.isEmpty())
        truthProperty = "Label";

    TemplateList predicted(TemplateList::fromGallery(predictedGallery));
    TemplateList truth(TemplateList::fromGallery(truthGallery));
    if (predicted.size() != truth.size()) qFatal("Input size mismatch.");

    QHash<QString, Counter> counters;
    for (int i=0; i<predicted.size(); i++) {
        if (predicted[i].file.name != truth[i].file.name)
            qFatal("Input order mismatch.");

        QString predictedSubject = predicted[i].file.get<QString>(predictedProperty);
        QString trueSubject = truth[i].file.get<QString>(truthProperty);

        QStringList predictedSubjects(predictedSubject);
        QStringList trueSubjects(trueSubject);

        foreach (const QString &subject, trueSubjects.toVector() /* Hack to copy the list. */) {
            if (predictedSubjects.contains(subject)) {
                counters[subject].truePositive++;
                trueSubjects.removeOne(subject);
                predictedSubjects.removeOne(subject);
            } else {
                counters[subject].falseNegative++;
            }
        }

        for (int i=0; i<trueSubjects.size(); i++)
            foreach (const QString &subject, predictedSubjects)
                counters[subject].falsePositive += 1.f / predictedSubjects.size();
    }

    const QStringList keys = counters.keys();
    QSharedPointer<Output> output(Output::make("", FileList() << "Count" << "Precision" << "Recall" << "F-score", FileList(keys)));

    int tpc = 0;
    int fnc = 0;

    for (int i=0; i<counters.size(); i++) {
        const QString &subject = keys[i];
        const Counter &counter = counters[subject];
        tpc += counter.truePositive;
        fnc += counter.falseNegative;
        const int count = counter.truePositive + counter.falseNegative;
        const float precision = counter.truePositive / (float)(counter.truePositive + counter.falsePositive);
        const float recall = counter.truePositive / (float)(counter.truePositive + counter.falseNegative);
        const float fscore = 2 * precision * recall / (precision + recall);
        output->setRelative(count, i, 0);
        output->setRelative(precision, i, 1);
        output->setRelative(recall, i, 2);
        output->setRelative(fscore, i, 3);
    }

    qDebug("Overall Accuracy = %f", (float)tpc / (float)(tpc + fnc));
}

struct Detection
{
    QRectF boundingBox;
    float confidence;

    Detection() {}
    Detection(const QRectF &boundingBox_, float confidence_ = -1)
        : boundingBox(boundingBox_), confidence(confidence_) {}

    float area() const { return boundingBox.width() * boundingBox.height(); }
    float overlap(const Detection &other) const
    {
        const Detection intersection(boundingBox.intersected(other.boundingBox));
        return intersection.area() / (area() + other.area() - intersection.area());
    }
};

struct Detections
{
    QList<Detection> predicted, truth;
};

struct ResolvedDetection
{
    float confidence, overlap;
    ResolvedDetection() : confidence(-1), overlap(-1) {}
    ResolvedDetection(float confidence_, float overlap_) : confidence(confidence_), overlap(overlap_) {}
    inline bool operator<(const ResolvedDetection &other) const { return confidence > other.confidence; }
};

struct DetectionOperatingPoint
{
    float Recall, FalsePositives, Precision;
    DetectionOperatingPoint() : Recall(-1), FalsePositives(-1), Precision(-1) {}
    DetectionOperatingPoint(float TP, float FP, float totalPositives)
        : Recall(TP/totalPositives), FalsePositives(FP), Precision(TP/(TP+FP)) {}
};

static QStringList computeDetectionResults(const QList<ResolvedDetection> &detections, int totalPositives, bool discrete)
{
    QList<DetectionOperatingPoint> points;
    float TP = 0, FP = 0, prevFP = -1;
    for (int i=0; i<detections.size(); i++) {
        const ResolvedDetection &detection = detections[i];
        if (discrete) {
            if (detection.overlap >= 0.5) TP++;
            else                          FP++;
        } else {
            TP += detection.overlap;
            FP += 1 - detection.overlap;
        }
        if ((i == detections.size()-1) || (detection.confidence > detections[i+1].confidence)) {
            if (FP > prevFP || (i == detections.size()-1)) {
                points.append(DetectionOperatingPoint(TP, FP, totalPositives));
                prevFP = FP;
            }
        }
    }

    const int keep = qMin(points.size(), Max_Points);
    if (keep < 1) qFatal("Insufficient points.");

    QStringList lines; lines.reserve(keep);
    if (keep == 1) {
        const DetectionOperatingPoint &point = points[0];
        lines.append(QString("%1ROC, %2, %3").arg(discrete ? "Discrete" : "Continuous", QString::number(point.FalsePositives), QString::number(point.Recall)));
        lines.append(QString("%1PR, %2, %3").arg(discrete ? "Discrete" : "Continuous", QString::number(point.Recall), QString::number(point.Precision)));
    } else {
        for (int i=0; i<keep; i++) {
            const DetectionOperatingPoint &point = points[double(i) / double(keep-1) * double(points.size()-1)];
            lines.append(QString("%1ROC, %2, %3").arg(discrete ? "Discrete" : "Continuous", QString::number(point.FalsePositives), QString::number(point.Recall)));
            lines.append(QString("%1PR, %2, %3").arg(discrete ? "Discrete" : "Continuous", QString::number(point.Recall), QString::number(point.Precision)));
        }
    }
    return lines;
}

QString getDetectKey(const TemplateList &templates)
{
    const File &f = templates.first().file;
    foreach (const QString &key, f.localKeys()) {
        // first check for single detections
        if (!f.get<QRectF>(key, QRectF()).isNull())
            return key;
    }
    // and then multiple
    if (!f.rects().empty())
        return "Rects";
    return "";
}

bool detectKeyIsList(QString key, const TemplateList &templates)
{
    return templates.first().file.get<QRectF>(key, QRectF()).isNull();
}

// return a list of detections whether the template holds
// multiple detections or a single detection
QList<Detection> getDetections(QString key, const Template &t, bool isList, bool isTruth)
{
    File f = t.file;
    QList<Detection> dets;
    if (isList) {
        // TODO: handle Confidence for multiple detections in a template
        foreach (const QRectF &rect, f.rects())
            dets.append(Detection(rect));
    } else {
        if (isTruth) {
            dets.append(Detection(f.get<QRectF>(key)));
        } else {
            dets.append(Detection(f.get<QRectF>(key), f.get<float>("Confidence", -1)));
        }
    }
    return dets;
}

QMap<QString, Detections> getDetections(const TemplateList &predicted, const TemplateList &truth)
{
    // Figure out which metadata field contains a bounding box
    QString truthDetectKey = getDetectKey(truth);
    if (truthDetectKey.isEmpty()) qFatal("No suitable ground truth metadata key found.");
    QString predictedDetectKey = getDetectKey(predicted);
    if (predictedDetectKey.isEmpty()) qFatal("No suitable predicted metadata key found.");
    qDebug("Using metadata key: %s%s",
           qPrintable(predictedDetectKey),
           qPrintable(predictedDetectKey == truthDetectKey ? QString() : "/"+truthDetectKey));

    QMap<QString, Detections> allDetections;
    bool predKeyIsList = detectKeyIsList(predictedDetectKey, predicted);
    bool truthKeyIsList = detectKeyIsList(truthDetectKey, truth);
    foreach (const Template &t, predicted) {
        QList<Detection> dets = getDetections(predictedDetectKey, t, predKeyIsList, false);
        allDetections[t.file.baseName()].predicted.append(dets);
    }
    foreach (const Template &t, truth) {
        QList<Detection> dets = getDetections(truthDetectKey, t, truthKeyIsList, true);
        allDetections[t.file.baseName()].truth.append(dets);
    }
    return allDetections;
}

float EvalDetection(const QString &predictedGallery, const QString &truthGallery, const QString &csv)
{
    qDebug("Evaluating detection of %s against %s", qPrintable(predictedGallery), qPrintable(truthGallery));
    const TemplateList predicted(TemplateList::fromGallery(predictedGallery));
    const TemplateList truth(TemplateList::fromGallery(truthGallery));

    // Organized by file, QMap used to preserve order
    QMap<QString, Detections> allDetections = getDetections(predicted, truth);

    QList<ResolvedDetection> resolvedDetections, falseNegativeDetections;
    foreach (Detections detections, allDetections.values()) {
        while (!detections.truth.isEmpty() && !detections.predicted.isEmpty()) {
            const Detection truth = detections.truth.takeFirst();
            int bestIndex = -1;
            float bestOverlap = -std::numeric_limits<float>::max();
            for (int i=0; i<detections.predicted.size(); i++) {
                const float overlap = truth.overlap(detections.predicted[i]);
                if (overlap > bestOverlap) {
                    bestOverlap = overlap;
                    bestIndex = i;
                }
            }
            const Detection predicted = detections.predicted.takeAt(bestIndex);
            resolvedDetections.append(ResolvedDetection(predicted.confidence, bestOverlap));
        }

        foreach (const Detection &detection, detections.predicted)
            resolvedDetections.append(ResolvedDetection(detection.confidence, 0));
        for (int i=0; i<detections.truth.size(); i++)
            falseNegativeDetections.append(ResolvedDetection(-std::numeric_limits<float>::max(), 0));
    }

    std::sort(resolvedDetections.begin(), resolvedDetections.end());

    QStringList lines;
    lines.append("Plot, X, Y");
    lines.append(computeDetectionResults(resolvedDetections, truth.size(), true));
    lines.append(computeDetectionResults(resolvedDetections, truth.size(), false));

    float averageOverlap;
    { // Overlap Density
        QList<ResolvedDetection> allDetections; allDetections << resolvedDetections << falseNegativeDetections;
        const int keep = qMin(allDetections.size(), Max_Points);
        lines.reserve(lines.size() + keep);
        float totalOverlap = 0;
        for (int i=0; i<keep; i++) {
            const float overlap = allDetections[double(i) / double(keep-1) * double(allDetections.size()-1)].overlap;
            totalOverlap += overlap;
            lines.append(QString("Overlap,%1,1").arg(QString::number(allDetections[double(i) / double(keep-1) * double(allDetections.size()-1)].overlap)));
        }
        averageOverlap = totalOverlap / keep;
    }
    lines.append(QString("AverageOverlap,%1,").arg(QString::number(averageOverlap)));

    QtUtils::writeFile(csv, lines);
    qDebug("Average Overlap = %.3f", averageOverlap);
    return averageOverlap;
}

float EvalLandmarking(const QString &predictedGallery, const QString &truthGallery, const QString &csv, int normalizationIndexA, int normalizationIndexB)
{
    qDebug("Evaluating landmarking of %s against %s", qPrintable(predictedGallery), qPrintable(truthGallery));
    const TemplateList predicted(TemplateList::fromGallery(predictedGallery));
    const TemplateList truth(TemplateList::fromGallery(truthGallery));
    const QStringList predictedNames = File::get<QString>(predicted, "name");
    const QStringList truthNames = File::get<QString>(truth, "name");

    QList< QList<float> > pointErrors;
    for (int i=0; i<predicted.size(); i++) {
        const QString &predictedName = predictedNames[i];
        const int truthIndex = truthNames.indexOf(predictedName);
        if (truthIndex == -1) qFatal("Could not identify ground truth for file: %s", qPrintable(predictedName));
        const QList<QPointF> predictedPoints = predicted[i].file.points();
        const QList<QPointF> truthPoints = truth[truthIndex].file.points();
        if (predictedPoints.size() != truthPoints.size()) qFatal("Points size mismatch for file: %s", qPrintable(predictedName));
        while (pointErrors.size() < predictedPoints.size())
            pointErrors.append(QList<float>());
        if (normalizationIndexA >= truthPoints.size()) qFatal("Normalization index A is out of range.");
        if (normalizationIndexB >= truthPoints.size()) qFatal("Normalization index B is out of range.");
        const float normalizedLength = QtUtils::euclideanLength(truthPoints[normalizationIndexB] - truthPoints[normalizationIndexA]);
        for (int j=0; j<predictedPoints.size(); j++)
            pointErrors[j].append(QtUtils::euclideanLength(predictedPoints[j] - truthPoints[j])/normalizedLength);
    }

    QList<float> averagePointErrors; averagePointErrors.reserve(pointErrors.size());
    for (int i=0; i<pointErrors.size(); i++) {
        std::sort(pointErrors[i].begin(), pointErrors[i].end());
        averagePointErrors.append(Common::Mean(pointErrors[i]));
    }
    const float averagePointError = Common::Mean(averagePointErrors);

    QStringList lines;
    lines.append("Plot,X,Y");
    for (int i=0; i<pointErrors.size(); i++) {
        const QList<float> &pointError = pointErrors[i];
        const int keep = qMin(Max_Points, pointError.size());
        for (int j=0; j<keep; j++)
            lines.append(QString("Box,%1,%2").arg(QString::number(i), QString::number(pointError[j*(pointError.size()-1)/(keep-1)])));
    }

    QtUtils::writeFile(csv, lines);
    qDebug("Average Error: %.3f", averagePointError);
    return averagePointError;
}

void EvalRegression(const QString &predictedGallery, const QString &truthGallery, QString predictedProperty, QString truthProperty)
{
    qDebug("Evaluating regression of %s against %s", qPrintable(predictedGallery), qPrintable(truthGallery));

    if (predictedProperty.isEmpty())
        predictedProperty = "Regressor";
    // If predictedProperty is specified, but truthProperty isn't, copy the value over
    // rather than using the default for truthProperty
    else if (truthProperty.isEmpty())
        truthProperty = predictedProperty;

    if (truthProperty.isEmpty())
        predictedProperty = "Regressand";

    const TemplateList predicted(TemplateList::fromGallery(predictedGallery));
    const TemplateList truth(TemplateList::fromGallery(truthGallery));
    if (predicted.size() != truth.size()) qFatal("Input size mismatch.");

    float rmsError = 0;
    float maeError = 0;
    QStringList truthValues, predictedValues;
    for (int i=0; i<predicted.size(); i++) {
        if (predicted[i].file.name != truth[i].file.name)
            qFatal("Input order mismatch.");

        float difference = predicted[i].file.get<float>(predictedProperty) - truth[i].file.get<float>(truthProperty);

        rmsError += pow(difference, 2.f);
        maeError += fabsf(difference);
        truthValues.append(QString::number(truth[i].file.get<float>(truthProperty)));
        predictedValues.append(QString::number(predicted[i].file.get<float>(predictedProperty)));
    }

    QStringList rSource;
    rSource << "# Load libraries" << "library(ggplot2)" << "" << "# Set Data"
            << "Actual <- c(" + truthValues.join(",") + ")"
            << "Predicted <- c(" + predictedValues.join(",") + ")"
            << "data <- data.frame(Actual, Predicted)"
            << "" << "# Construct Plot" << "pdf(\"EvalRegression.pdf\")"
            << "print(qplot(Actual, Predicted, data=data, geom=\"jitter\", alpha=I(2/3)) + geom_abline(intercept=0, slope=1, color=\"forestgreen\", size=I(1)) + geom_smooth(size=I(1), color=\"mediumblue\") + theme_bw())"
            << "print(qplot(Actual, Predicted-Actual, data=data, geom=\"jitter\", alpha=I(2/3)) + geom_abline(intercept=0, slope=0, color=\"forestgreen\", size=I(1)) + geom_smooth(size=I(1), color=\"mediumblue\") + theme_bw())"
            << "dev.off()";


    QString rFile = "EvalRegression.R";
    QtUtils::writeFile(rFile, rSource);
    bool success = QtUtils::runRScript(rFile);
    if (success) QtUtils::showFile("EvalRegression.pdf");

    qDebug("RMS Error = %f", sqrt(rmsError/predicted.size()));
    qDebug("MAE = %f", maeError/predicted.size());
}

} // namespace br
