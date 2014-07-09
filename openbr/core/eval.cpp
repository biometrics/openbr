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
#include <QMapIterator>

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

// Decide whether to construct a normal mask matrix, or a pairwise mask by comparing the dimensions of
// scores with the size of the target and query lists
static cv::Mat constructMatchingMask(const cv::Mat &scores, const FileList &target, const FileList &query, int partition=0)
{
    // If the dimensions of the score matrix match the sizes of the target and query lists, construct a normal mask matrix
    if (target.size() == scores.cols && query.size() == scores.rows)
        return BEE::makeMask(target, query, partition);
    // If this looks like a pairwise comparison (1 column score matrix, equal length target and query sets), construct a
    // mask for that
    else if (scores.cols == 1 && target.size() == query.size()) {
        return BEE::makePairwiseMask(target, query, partition);
    }
    // otherwise, we fail
    else
        qFatal("Unable to construct mask for %d by %d score matrix from %d element query set, and %d element target set ", scores.rows, scores.cols, query.length(), target.length());
}

float Evaluate(const cv::Mat &scores, const FileList &target, const FileList &query, const QString &csv, int partition)
{
    return Evaluate(scores, constructMatchingMask(scores, target, query, partition), csv);
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
        scores = BEE::readMatrix(simmat, &target, &query);
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

        truth = constructMatchingMask(scores, TemplateList::fromGallery(target).files(),
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

    if (simmat.type() != CV_32FC1)
        qFatal("Invalid simmat format");

    if (mask.type() != CV_8UC1)
        qFatal("Invalid mask format");

    float result = -1;

    // Make comparisons
    QList<Comparison> comparisons; comparisons.reserve(simmat.rows*simmat.cols);
    int genuineCount = 0, impostorCount = 0, numNaNs = 0;
    for (int i=0; i<simmat.rows; i++) {
        for (int j=0; j<simmat.cols; j++) {
            const BEE::MaskValue mask_val = mask.at<BEE::MaskValue>(i,j);
            const BEE::SimmatValue simmat_val = simmat.at<BEE::SimmatValue>(i,j);
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

struct GenImpCounts
{
    GenImpCounts()
    {
        genCount = 1;
        impCount = 0;
    }

    qint64 genCount;
    qint64 impCount;
};

float InplaceEval(const QString &simmat, const QString &target, const QString &query, const QString &csv)
{
    qDebug("Evaluating %s%s%s",
            qPrintable(simmat),
            qPrintable(" with " + target + " and " + query),
            csv.isEmpty() ? "" : qPrintable(" to " + csv));

    // To start with, we will find the size of the header, and check if the file size is consistent with the information
    // given in the header.
    QFile file(simmat);
    bool success = file.open(QFile::ReadOnly);
    if (!success) qFatal("Unable to open %s for reading.", qPrintable(simmat));

    // Check format
    QByteArray format = file.readLine();
    if (format[1] != '2') qFatal("Invalid matrix header.");

    // Read sigset names, we dont' care if they are valid, just want to advance the file pointer.
    file.readLine();
    file.readLine();

    // Get matrix size
    QStringList words = QString(file.readLine()).split(" ");
    qint64 rows = words[1].toLongLong();
    qint64 cols = words[2].toLongLong();

    bool isMask = words[0][1] == 'B';
    qint64 typeSize = isMask ? sizeof(BEE::MaskValue) : sizeof(BEE::SimmatValue);

    // Get matrix data
    qint64 rowSize = cols * typeSize;

    // after reading the header, we are at the start of the matrix data
    qint64 data_pos = file.pos();

    // Map each unique label to a list of positions in the gallery
    QMap<QString, QList<qint64> > galleryIndices;

    // Next we will find the locations of all genuine scores based on the galleries, we will not instantiate a mask matrix
    QScopedPointer<Gallery> columnGal(Gallery::make(target));
    columnGal->set_readBlockSize(10000);

    qint64 idx  = 0;
    bool done = false;
    do {
        TemplateList temp = columnGal->readBlock(&done);
        QStringList tempLabels = File::get<QString>(temp, "Label");

        foreach (QString st, tempLabels) {
            if (!galleryIndices.contains(st))
                galleryIndices.insert(st, QList<qint64>());

            galleryIndices[st].append(idx);
            idx++;
        }
    } while (!done);

    qint64 genTotal = 0;
    qint64 imposterTotal = 0;

    // map a genuine score threshold to the set of imposter scores uniquely rejected at that threshold
    QMap<float, GenImpCounts> genScoresToCounts;

    QScopedPointer<Gallery> probeGallery (Gallery::make(query));
    probeGallery->set_readBlockSize(10000);
    done = false;
    qint64 row_count = 0;
    do {
        TemplateList temp = probeGallery->readBlock(&done);
        QStringList probeLabels = File::get<QString>(temp, "Label");

        for (int i=0; i < probeLabels.size();i++) {
            row_count++;
            if (!galleryIndices.contains(probeLabels[i]))
                continue;

            QList<qint64> colMask = galleryIndices[probeLabels[i]];
            foreach (qint64 colID, colMask) {
                float score;
                file.seek(data_pos + i * rowSize + colID * typeSize);
                file.read((char *) &score, sizeof(float));
                if (genScoresToCounts.contains(score))
                    genScoresToCounts[score].genCount++;
                else
                    genScoresToCounts.insert(score, GenImpCounts());
                genTotal++;
            }
        }

    } while (!done);

    QMap<float, GenImpCounts> noImpostors = genScoresToCounts;

    imposterTotal = rows * cols - genTotal;

    file.seek(data_pos);
    cv::Mat aRow(1, cols, CV_32FC1);
    qint64 highImpostors = 0;

    QScopedPointer<Gallery> probeGallery2 (Gallery::make(query));
    int bSize = 10000;
    probeGallery2->set_readBlockSize(bSize);
    done = false;
    row_count  = 0;

    //sequence, mapfunciton, reducefunction
    Mat blockMat(bSize, cols, CV_32FC1);

    qint64 bCount = 0;
    do {
        bCount++;
        TemplateList temp = probeGallery2->readBlock(&done);
        QStringList probeLabels = File::get<QString>(temp, "Label");
        temp.clear();

        file.read((char *) blockMat.data, rowSize * probeLabels.length());
        for (int i=0; i < probeLabels.size();i++) {
            row_count++;
            aRow = blockMat.row(i);

            QList<qint64> colMask = galleryIndices[probeLabels[i]];
            int listIdx = 0;

            for (qint64 colIdx = 0; colIdx < cols; colIdx++)
            {
                // if our list index is past the end of colMask, we just have impostor scores left
                if (listIdx < colMask.size() )
                {
                    // we hit the next gen score, skip it, and advance listIdx
                    if (colIdx == colMask[listIdx])
                    {
                        listIdx++;
                        continue;
                    }
                }
                float score = aRow.at<float>(0, colIdx);
                QMap<float, GenImpCounts>::iterator i = genScoresToCounts.upperBound(score);
                if (i == genScoresToCounts.end() )
                {
                    // no genuine scores >= this impostor score, nothing to do.
                    highImpostors++;
                    continue;
                }

                // The iterator points to the first score > this one, i.e. the highest threshold for which this
                // score will be rejected
                i->impCount++;
            }
        }
    } while (!done);

    QList<OperatingPoint> operatingPoints;
    qint64 genAccum = 0;
    qint64 impAccum = highImpostors;

    QMapIterator<float, GenImpCounts> i(genScoresToCounts);
    
    i.toBack();
   
    // iterating in reverse order of thresholds
    while (i.hasPrevious()) {
        i.previous();
        // we want to accumulate false accept, true accept points
        float thresh = i.key();
        // genAccum -- number of gen scores at this threshold and above
        genAccum += i.value().genCount;

        operatingPoints.append(OperatingPoint(thresh, float(impAccum) / float(imposterTotal), float(genAccum) / float(genTotal)));

        // imp count -- number of impostor scores at this threshold and above
        impAccum += i.value().impCount;
    }

    QStringList lines;
    lines.append("Plot,X,Y");
    lines.append("Metadata,"+QString::number(cols)+",Gallery");
    lines.append("Metadata,"+QString::number(rows)+",Probe");
    lines.append("Metadata,"+QString::number(genTotal)+",Genuine");
    lines.append("Metadata,"+QString::number(imposterTotal)+",Impostor");
    lines.append("Metadata,"+QString::number(cols*rows-(genTotal+imposterTotal))+",Ignored");

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

    float result;
    // Write FAR/TAR Bar Chart (BC)
    lines.append(qPrintable(QString("BC,0.001,%1").arg(QString::number(getTAR(operatingPoints, 0.001), 'f', 3))));
    lines.append(qPrintable(QString("BC,0.01,%1").arg(QString::number(result = getTAR(operatingPoints, 0.01), 'f', 3))));

    qDebug("TAR @ FAR = 0.01: %.3f", result);
    QtUtils::writeFile(csv, lines);
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

static QStringList computeDetectionResults(const QList<ResolvedDetection> &detections, int totalTrueDetections, bool discrete)
{
    QList<DetectionOperatingPoint> points;
    float TP = 0, FP = 0, prevFP = -1;
    for (int i=0; i<detections.size(); i++) {
        const ResolvedDetection &detection = detections[i];
        if (discrete) {
            // A 50% overlap is considered a true positive
            if (detection.overlap >= 0.5) TP++;
            else                          FP++;
        } else {
            TP += detection.overlap;
            FP += 1 - detection.overlap;
        }
        if ((i == detections.size()-1) || (detection.confidence > detections[i+1].confidence)) {
            if (FP > prevFP || (i == detections.size()-1)) {
                points.append(DetectionOperatingPoint(TP, FP, totalTrueDetections));
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

static DetectionKey getDetectKey(const FileList &files)
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
static QList<Detection> getDetections(const DetectionKey &key, const File &f, bool isTruth)
{
    QList<Detection> dets;
    if (key.type == DetectionKey::RectList) {
        QList<QRectF> rects = f.rects();
        QList<float> confidences = f.getList<float>("Confidences", QList<float>());
        if (!isTruth && rects.size() != confidences.size())
            qFatal("You don't have enough confidence. I mean, your detections don't all have confidence measures.");
        for (int i=0; i<rects.size(); i++) {
            if (isTruth)
                dets.append(Detection(rects[i]));
            else
                dets.append(Detection(rects[i], confidences[i]));
        }
    } else if (key.type == DetectionKey::Rect) {
        dets.append(Detection(f.get<QRectF>(key), isTruth ? -1 : f.get<float>("Confidence", -1)));
    } else if (key.type == DetectionKey::XYWidthHeight) {
        const QRectF rect(f.get<float>(key+"_X"), f.get<float>(key+"_Y"), f.get<float>(key+"_Width"), f.get<float>(key+"_Height"));
        dets.append(Detection(rect, isTruth ? -1 : f.get<float>("Confidence", -1)));
    }
    return dets;
}

static QMap<QString, Detections> getDetections(const File &predictedGallery, const File &truthGallery)
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

    qDebug("Using metadata key: %s%s",
           qPrintable(predictedDetectKey),
           qPrintable(predictedDetectKey == truthDetectKey ? QString() : "/"+truthDetectKey));

    QMap<QString, Detections> allDetections;
    foreach (const File &f, predicted)
        allDetections[f.baseName()].predicted.append(getDetections(predictedDetectKey, f, false));
    foreach (const File &f, truth)
        allDetections[f.baseName()].truth.append(getDetections(truthDetectKey, f, true));
    return allDetections;
}

static int associateGroundTruthDetections(QList<ResolvedDetection> &resolved, QList<ResolvedDetection> &falseNegative, QMap<QString, Detections> &all, QRectF &offsets)
{
    float dLeftTotal = 0.0, dRightTotal = 0.0, dTopTotal = 0.0, dBottomTotal = 0.0;
    int count = 0, totalTrueDetections = 0;

    foreach (Detections detections, all.values()) {
        totalTrueDetections += detections.truth.size();
        // Try to associate ground truth detections with predicted detections
        while (!detections.truth.isEmpty() && !detections.predicted.isEmpty()) {
            const Detection truth = detections.truth.takeFirst(); // Take removes the detection
            int bestIndex = -1;
            float bestOverlap = -std::numeric_limits<float>::max();
            // Find the nearest predicted detection to this ground truth detection
            for (int i=0; i<detections.predicted.size(); i++) {
                Detection predicted = detections.predicted[i];
                float predictedWidth = predicted.boundingBox.width();
                float x, y, width, height;
                x = predicted.boundingBox.x() + offsets.x()*predictedWidth;
                y = predicted.boundingBox.y() + offsets.y()*predictedWidth;
                width = predicted.boundingBox.width() - offsets.width()*predictedWidth;
                height = predicted.boundingBox.height() - offsets.height()*predictedWidth;
                Detection newPredicted(QRectF(x, y, width, height), 0.0);

                const float overlap = truth.overlap(newPredicted);
                if (overlap > bestOverlap) {
                    bestOverlap = overlap;
                    bestIndex = i;
                }
            }
            // Removing the detection prevents us from considering it twice.
            // We don't want to associate two ground truth detections with the
            // same prediction, over vice versa.
            const Detection predicted = detections.predicted.takeAt(bestIndex);
            resolved.append(ResolvedDetection(predicted.confidence, bestOverlap));

            if (offsets.x() == 0) {
                // Add side differences to total only for pairs that meet the overlap threshold.
                if (bestOverlap > 0.3) {
                    count++;
                    float width = predicted.boundingBox.width();
                    dLeftTotal += (truth.boundingBox.left() - predicted.boundingBox.left()) / width;
                    dRightTotal += (truth.boundingBox.right() - predicted.boundingBox.right()) / width;
                    dTopTotal += (truth.boundingBox.top() - predicted.boundingBox.top()) / width;
                    dBottomTotal += (truth.boundingBox.bottom() - predicted.boundingBox.bottom()) / width;
                }
            }
        }

        foreach (const Detection &detection, detections.predicted)
            resolved.append(ResolvedDetection(detection.confidence, 0));
        for (int i=0; i<detections.truth.size(); i++)
            falseNegative.append(ResolvedDetection(-std::numeric_limits<float>::max(), 0));
    }
    if (offsets.x() == 0) {
        // Calculate average differences in each direction
        float dRight = dRightTotal / count;
        float dBottom = dBottomTotal / count;
        float dX = dLeftTotal / count;
        float dY = dTopTotal / count;
        float dWidth = dX - dRight;
        float dHeight = dY - dBottom;

        offsets.setX(dX);
        offsets.setY(dY);
        offsets.setWidth(dWidth);
        offsets.setHeight(dHeight);
    }
    return totalTrueDetections;
}

float EvalDetection(const QString &predictedGallery, const QString &truthGallery, const QString &csv, bool normalize)
{
    qDebug("Evaluating detection of %s against %s", qPrintable(predictedGallery), qPrintable(truthGallery));
    // Organized by file, QMap used to preserve order
    QMap<QString, Detections> allDetections = getDetections(predictedGallery, truthGallery);

    QList<ResolvedDetection> resolvedDetections, falseNegativeDetections;
    QRectF normalizations(0, 0, 0, 0);
    
    // Associate predictions to ground truth
    int totalTrueDetections = associateGroundTruthDetections(resolvedDetections, falseNegativeDetections, allDetections, normalizations);

    // Redo association of ground truth to predictions with boundingBoxes
    // resized based on the average differences on each side.
    if (normalize) {
        qDebug("dX = %.3f", normalizations.x());
        qDebug("dY = %.3f", normalizations.y());
        qDebug("dWidth = %.3f", normalizations.width());
        qDebug("dHeight = %.3f", normalizations.height());
        resolvedDetections.clear();
        falseNegativeDetections.clear();
        totalTrueDetections = associateGroundTruthDetections(resolvedDetections, falseNegativeDetections, allDetections, normalizations);
    }
    std::sort(resolvedDetections.begin(), resolvedDetections.end());
    QStringList lines;
    lines.append("Plot, X, Y");
    lines.append(computeDetectionResults(resolvedDetections, totalTrueDetections, true));
    lines.append(computeDetectionResults(resolvedDetections, totalTrueDetections, false));

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

    int skipped = 0;
    QList< QList<float> > pointErrors;
    for (int i=0; i<predicted.size(); i++) {
        const QString &predictedName = predictedNames[i];
        const int truthIndex = truthNames.indexOf(predictedName);
        if (truthIndex == -1) qFatal("Could not identify ground truth for file: %s", qPrintable(predictedName));
        const QList<QPointF> predictedPoints = predicted[i].file.points();
        const QList<QPointF> truthPoints = truth[truthIndex].file.points();
        if (predictedPoints.size() != truthPoints.size()) {
            skipped++;
            continue;
        }
        while (pointErrors.size() < predictedPoints.size())
            pointErrors.append(QList<float>());
        if (normalizationIndexA >= truthPoints.size()) qFatal("Normalization index A is out of range.");
        if (normalizationIndexB >= truthPoints.size()) qFatal("Normalization index B is out of range.");
        const float normalizedLength = QtUtils::euclideanLength(truthPoints[normalizationIndexB] - truthPoints[normalizationIndexA]);
        for (int j=0; j<predictedPoints.size(); j++)
            pointErrors[j].append(QtUtils::euclideanLength(predictedPoints[j] - truthPoints[j])/normalizedLength);
    }
    qDebug() << "Skipped " << skipped << " files due to point size mismatch.";

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

    lines.append(QString("AvgError,0,%1").arg(averagePointError));

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
