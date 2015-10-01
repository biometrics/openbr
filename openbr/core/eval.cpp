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
#include "openbr/core/opencvutils.h"
#include <QMapIterator>
#include <cmath>

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

    inline bool operator<(const Comparison &other) const
    {
        if (score != other.score) return (score > other.score);
        else                      return !genuine && other.genuine; // Tie-break favors pessimistic behavior of ranking impostors higher.
    }
};

#undef FAR // Windows preprecessor definition conflicts with variable name
struct OperatingPoint
{
    float score, FAR, TAR;
    OperatingPoint() {}
    OperatingPoint(float _score, float _FAR, float _TAR)
        : score(_score), FAR(_FAR), TAR(_TAR) {}
};

static OperatingPoint getOperatingPointGivenFAR(const QList<OperatingPoint> &operatingPoints, float FAR)
{
    int index = 0;
    while (operatingPoints[index].FAR < FAR) {
        index++;
        if (index == operatingPoints.size())
            return OperatingPoint(operatingPoints.last().score, FAR, operatingPoints.last().TAR);
    }

    const float FAR1 = (index == 0 ? 0 : operatingPoints[index-1].FAR);
    const float TAR1 = (index == 0 ? 0 : operatingPoints[index-1].TAR);
    const float score1 = (index == 0 ? operatingPoints[index].score : operatingPoints[index-1].score);
    const float FAR2 = operatingPoints[index].FAR;
    const float TAR2 = operatingPoints[index].TAR;
    const float score2 = operatingPoints[index].score;
    const float mTAR = (TAR2 - TAR1) / (FAR2 - FAR1);
    const float bTAR = TAR1 - mTAR*FAR1;
    const float mScore = (score2 - score1) / (FAR2 - FAR1);
    const float bScore = score1 - mScore*FAR1;
    return OperatingPoint(mScore * FAR + bScore, FAR, mTAR * FAR + bTAR);
}

static OperatingPoint getOperatingPointGivenTAR(const QList<OperatingPoint> &operatingPoints, float TAR)
{
    int index = 0;
    while (operatingPoints[index].TAR < TAR) {
      index++;
      if (index == operatingPoints.size())
            return OperatingPoint(operatingPoints.last().score, operatingPoints.last().FAR, TAR);
    }

    const float FAR1 = (index == 0 ? 0 : operatingPoints[index-1].FAR);
    const float TAR1 = (index == 0 ? 0 : operatingPoints[index-1].TAR);
    const float score1 = (index == 0 ? operatingPoints[index].score : operatingPoints[index-1].score);
    const float FAR2 = operatingPoints[index].FAR;
    const float TAR2 = operatingPoints[index].TAR;
    const float score2 = operatingPoints[index].score;
    const float mTAR = (TAR2 - TAR1) / (FAR2 - FAR1);
    const float bTAR = TAR1 - mTAR*FAR1;
    const float mScore = (score2 - score1) / (FAR2 - FAR1);
    const float bScore = score1 - mScore*FAR1;

    const float FAR = (TAR - bTAR) / mTAR;
    return OperatingPoint(mScore * FAR + bScore, FAR, TAR);
}


static float getCMC(const QVector<int> &firstGenuineReturns, int rank, size_t possibleReturns = 0)
{
    bool calcPossible = possibleReturns ? false : true;
    int realizedReturns = 0;
    foreach (int firstGenuineReturn, firstGenuineReturns) {
        if (firstGenuineReturn > 0) {
            if (calcPossible)
                possibleReturns++;
            if (firstGenuineReturn <= rank) realizedReturns++;
        }
    }
    const float retrievalRate = float(realizedReturns)/possibleReturns;
    return retrievalRate;
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

    return cv::Mat();
}

float Evaluate(const cv::Mat &scores, const FileList &target, const FileList &query, const File &csv, int partition)
{
    return Evaluate(scores, constructMatchingMask(scores, target, query, partition), csv, QString(), QString(), 0);
}

float Evaluate(const QString &simmat, const QString &mask, const File &csv, unsigned int matches)
{
    qDebug("Evaluating %s%s%s",
           qPrintable(simmat),
           mask.isEmpty() ? "" : qPrintable(" with " + mask),
           csv.name.isEmpty() ? "" : qPrintable(" to " + csv));

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

    return Evaluate(scores, truth, csv, target, query, matches);
}

float Evaluate(const Mat &simmat, const Mat &mask, const File &csv, const QString &target, const QString &query, unsigned int matches)
{
    if (target.isEmpty() || query.isEmpty()) matches = 0;
    if (simmat.size() != mask.size())
        qFatal("Similarity matrix (%ix%i) differs in size from mask matrix (%ix%i).",
               simmat.rows, simmat.cols, mask.rows, mask.cols);

    if (simmat.type() != CV_32FC1)
        qFatal("Invalid simmat format");

    if (mask.type() != CV_8UC1)
        qFatal("Invalid mask format");

    float result = -1;

    // Make comparisons
    QVector<Comparison> comparisons; comparisons.reserve(simmat.rows*simmat.cols);

    // Flags rows as being mated or non-mated searches
    // Positive value: mated search, negative value: non-mated search
    // Value of 0: ignored search
    QVector<int> genuineSearches(simmat.rows, 0);

    int totalGenuineSearches = 0, totalImpostorSearches = 0;
    int genuineCount = 0, impostorCount = 0, numNaNs = 0;
    for (int i=0; i<simmat.rows; i++) {
        for (int j=0; j<simmat.cols; j++) {
            const BEE::MaskValue mask_val = mask.at<BEE::MaskValue>(i,j);
            const BEE::SimmatValue simmat_val = simmat.at<BEE::SimmatValue>(i,j);
            if (mask_val == BEE::DontCare) continue;
            if (simmat_val != simmat_val) { numNaNs++; continue; }
            Comparison comparison(simmat_val, j, i, mask_val == BEE::Match);
            comparisons.append(comparison);
            if (comparison.genuine) {
                if (genuineSearches[comparison.query] != 1) {
                    genuineSearches[comparison.query] = 1;
                    totalGenuineSearches++;
                }
                genuineCount++;
            } else {
                if (genuineSearches[comparison.query] != 1) {
                    genuineSearches[comparison.query] = -1;
                }
                impostorCount++;
            }                           
        }
    }

    // This is not necessarily equal to (simmat.rows-totalGenuineSearches)
    // since some rows could consist entirely of ignored values
    foreach (int i, genuineSearches)
        if (i<0) totalImpostorSearches++;

    if (numNaNs > 0) qWarning("Encountered %d NaN scores!", numNaNs);
    if (genuineCount == 0) qFatal("No genuine scores!");
    if (impostorCount == 0) qFatal("No impostor scores!");

    // Sort comparisons by simmat_val (score)
    std::sort(comparisons.begin(), comparisons.end());

    QList<OperatingPoint> operatingPoints;
    QList<OperatingPoint> searchOperatingPoints;
    QList<float> genuines; genuines.reserve(sqrt((float)comparisons.size()));
    QList<float> impostors; impostors.reserve(comparisons.size());
    QVector<int> firstGenuineReturns(simmat.rows, 0);

    int falsePositives = 0, previousFalsePositives = 0;
    int truePositives = 0, previousTruePositives = 0;
    int falseSearches = 0, previousFalseSearches = 0;
    int trueSearches = 0, previousTrueSearches = 0;
    int index = 0;
    int EERIndex = 0;
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
                if (genuineSearches[comparison.query] == 1) {
                    genuineSearches[comparison.query]++;
                    // True positive identification
                    trueSearches++;
                }
                genuines.append(comparison.score);
                if (firstGenuineReturns[comparison.query] < 1)
                    firstGenuineReturns[comparison.query] = abs(firstGenuineReturns[comparison.query]) + 1;
                if ((comparison.score != -std::numeric_limits<float>::max()) &&
                    (comparison.score < minGenuineScore))
                    minGenuineScore = comparison.score;
            } else {
                falsePositives++;
                if (genuineSearches[comparison.query] == -1) {
                    genuineSearches[comparison.query]--;
                    // False positive identification
                    falseSearches++;
                }
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

            if (EERIndex == 0) {
                if (floor(float(falsePositives)/impostorCount*100+0.5)/100 == floor((1-float(truePositives)/genuineCount)*100+0.5)/100) EERIndex = index-1;
            }
            previousFalsePositives = falsePositives;
            previousTruePositives = truePositives;
        }

        if ((falseSearches > previousFalseSearches) &&
             (trueSearches > previousTrueSearches)) {
            searchOperatingPoints.append(OperatingPoint(thresh, float(falseSearches)/totalImpostorSearches, float(trueSearches)/totalGenuineSearches));
            previousFalseSearches = falseSearches;
            previousTrueSearches = trueSearches;
        }
    }

    if (operatingPoints.size() == 0) operatingPoints.append(OperatingPoint(1, 1, 1));
    if (operatingPoints.size() == 1) operatingPoints.prepend(OperatingPoint(0, 0, 0));
    if (operatingPoints.size() > 2)  operatingPoints.takeLast(); // Remove point (1,1)

    if (searchOperatingPoints.size() == 0) searchOperatingPoints.append(OperatingPoint(1, 1, 1));
    if (searchOperatingPoints.size() == 1) searchOperatingPoints.prepend(OperatingPoint(0, 0, 0));
    if (searchOperatingPoints.size() > 2)  searchOperatingPoints.takeLast();

    // Write Metadata table
    QStringList lines;
    lines.append("Plot,X,Y");
    lines.append("Metadata,"+QString::number(simmat.cols)+",Gallery");
    lines.append("Metadata,"+QString::number(simmat.rows)+",Probe");
    lines.append("Metadata,"+QString::number(genuineCount)+",Genuine");
    lines.append("Metadata,"+QString::number(impostorCount)+",Impostor");
    lines.append("Metadata,"+QString::number(simmat.cols*simmat.rows-(genuineCount+impostorCount))+",Ignored");

    QString filePath = Globals->path;
    if (matches != 0 && EERIndex != 0) {
        const FileList targetFiles = TemplateList::fromGallery(target).files();
        const FileList queryFiles = TemplateList::fromGallery(query).files();
        unsigned int count = 0;
        for (int i = EERIndex-1; i >= 0; i--) {
            if (!comparisons[i].genuine) {
                lines.append("IM,"+QString::number(comparisons[i].score)+","+targetFiles[comparisons[i].target].get<QString>("Label")+":"
                    +filePath+"/"+targetFiles[comparisons[i].target].name+":"+queryFiles[comparisons[i].query].get<QString>("Label")+":"+filePath+"/"+queryFiles[comparisons[i].query].name);
                if (++count == matches) break;
            }
        }
        count = 0;
        for (int i = EERIndex+1; i < comparisons.size(); i++) {
            if (comparisons[i].genuine) {
                lines.append("GM,"+QString::number(comparisons[i].score)+","+targetFiles[comparisons[i].target].get<QString>("Label")+":"
                    +filePath+"/"+targetFiles[comparisons[i].target].name+":"+queryFiles[comparisons[i].query].get<QString>("Label")+":"+filePath+"/"+queryFiles[comparisons[i].query].name);
                if (++count == matches) break;
            }
        }
    }

    // Write Detection Error Tradeoff (DET), PRE, REC, Identification Error Tradeoff (IET)
    float expFAR = csv.get<float>("FAR", std::max(ceil(log10(impostorCount)), 1.0));
    float expFRR = csv.get<float>("FRR", std::max(ceil(log10(genuineCount)), 1.0));
    float expFPIR = csv.get<float>("FPIR", std::max(ceil(log10(totalImpostorSearches)), 1.0));

    float FARstep = expFAR / (float)(Max_Points - 1);
    float FRRstep = expFRR / (float)(Max_Points - 1);
    float FPIRstep = expFPIR / (float)(Max_Points - 1);

    for (int i=0; i<Max_Points; i++) {
        float FAR = pow(10, -expFAR + i*FARstep);
        float FRR = pow(10, -expFRR + i*FRRstep);
        float FPIR = pow(10, -expFPIR + i*FPIRstep);

        OperatingPoint operatingPointFAR = getOperatingPointGivenFAR(operatingPoints, FAR);
        OperatingPoint operatingPointTAR = getOperatingPointGivenTAR(operatingPoints, 1-FRR);
        OperatingPoint searchOperatingPoint = getOperatingPointGivenFAR(searchOperatingPoints, FPIR);
        lines.append(QString("DET,%1,%2").arg(QString::number(FAR),
                                              QString::number(1-operatingPointFAR.TAR)));
        lines.append(QString("FAR,%1,%2").arg(QString::number(operatingPointFAR.score),
                                              QString::number(FAR)));
        lines.append(QString("FRR,%1,%2").arg(QString::number(operatingPointTAR.score),
                                              QString::number(FRR)));
        lines.append(QString("IET,%1,%2").arg(QString::number(searchOperatingPoint.FAR),
                                              QString::number(1-searchOperatingPoint.TAR)));
    }

    // Write TAR@FAR Table (TF)
    foreach (float far, QList<float>() << 1e-6 << 1e-5 << 1e-4 << 1e-3 << 1e-2 << 1e-1)
      lines.append(qPrintable(QString("TF,%1,%2").arg(
						      QString::number(far, 'f'),
						      QString::number(getOperatingPointGivenFAR(operatingPoints, far).TAR, 'f', 3))));

    // Write FAR@TAR Table (FT)
    foreach (float tar, QList<float>() << 0.4 << 0.5 << 0.65 << 0.75 << 0.85 << 0.95)
      lines.append(qPrintable(QString("FT,%1,%2").arg(
                         QString::number(tar, 'f', 2),
                         QString::number(getOperatingPointGivenTAR(operatingPoints, tar).FAR, 'f', 3))));

    //Write CMC Table (CT)
    lines.append(qPrintable(QString("CT,1,%1").arg(QString::number(getCMC(firstGenuineReturns, 1), 'f', 3))));
    lines.append(qPrintable(QString("CT,5,%1").arg(QString::number(getCMC(firstGenuineReturns, 5), 'f', 3))));
    lines.append(qPrintable(QString("CT,10,%1").arg(QString::number(getCMC(firstGenuineReturns, 10), 'f', 3))));
    lines.append(qPrintable(QString("CT,20,%1").arg(QString::number(getCMC(firstGenuineReturns, 20), 'f', 3))));
    lines.append(qPrintable(QString("CT,50,%1").arg(QString::number(getCMC(firstGenuineReturns, 50), 'f', 3))));
    lines.append(qPrintable(QString("CT,100,%1").arg(QString::number(getCMC(firstGenuineReturns, 100), 'f', 3))));

    // Write FAR/TAR Bar Chart (BC)
    lines.append(qPrintable(QString("BC,0.001,%1").arg(QString::number(getOperatingPointGivenFAR(operatingPoints, 0.001).TAR, 'f', 3))));
    lines.append(qPrintable(QString("BC,0.01,%1").arg(QString::number(result = getOperatingPointGivenFAR(operatingPoints, 0.01).TAR, 'f', 3))));

    // Attempt to read template size from enrolled gallery and write to output CSV
    size_t maxSize(0);
    if (target.endsWith(".gal") && QFileInfo(target).exists()) {
        foreach (const Template &t, TemplateList::fromGallery(target)) maxSize = max(maxSize, t.bytes());
        lines.append(QString("TS,,%1").arg(QString::number(maxSize)));
    }

    // Write SD & KDE
    int points = qMin(qMin(Max_Points, genuines.size()), impostors.size());
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
    for (int i=1; i<=Max_Retrieval; i++) {
        const float retrievalRate = getCMC(firstGenuineReturns, i);
        lines.append(qPrintable(QString("CMC,%1,%2").arg(QString::number(i), QString::number(retrievalRate))));
    }

    QtUtils::writeFile(csv, lines);
    if (maxSize > 0) qDebug("Template Size: %i bytes", (int)maxSize);
    qDebug("TAR @ FAR = 0.01:    %.3f",getOperatingPointGivenFAR(operatingPoints, 0.01).TAR);
    qDebug("TAR @ FAR = 0.001:   %.3f",getOperatingPointGivenFAR(operatingPoints, 0.001).TAR);
    qDebug("TAR @ FAR = 0.0001:  %.3f",getOperatingPointGivenFAR(operatingPoints, 0.0001).TAR);
    qDebug("TAR @ FAR = 0.00001: %.3f",getOperatingPointGivenFAR(operatingPoints, 0.00001).TAR);

    qDebug("FNIR @ FPIR = 0.1:   %.3f", 1-getOperatingPointGivenFAR(searchOperatingPoints, 0.1).TAR);
    qDebug("FNIR @ FPIR = 0.01:  %.3f", 1-getOperatingPointGivenFAR(searchOperatingPoints, 0.01).TAR);

    qDebug("\nRetrieval Rate @ Rank = %d: %.3f", Report_Retrieval, getCMC(firstGenuineReturns, Report_Retrieval));

    return result;
}

void assertEval(const QString &simmat, const QString &mask, float accuracy)
{
    float result = Evaluate(simmat, mask, "", 0);
    // Round result to nearest thousandth for comparison against input accuracy.  Input is expected to be from previous
    // results of br -eval.
    result = floor(result*1000+0.5)/1000;
    if (result < accuracy) {
        qFatal("TAR @ FAR = 0.01 does not meet required accuracy: %.3f < %.3f", result, accuracy);
    }
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
    lines.append(qPrintable(QString("BC,0.001,%1").arg(QString::number(getOperatingPointGivenFAR(operatingPoints, 0.001).TAR, 'f', 3))));
    lines.append(qPrintable(QString("BC,0.01,%1").arg(QString::number(result = getOperatingPointGivenFAR(operatingPoints, 0.01).TAR, 'f', 3))));

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
    // The ignore flag is useful when certain faces in an image should be ignored
    // and should not effect detection performance. Predicted detections that overlap
    // with an ignored truth detection will not count as a true positive, false positive,
    // true negative, or false negative, it will simply be ignored.
    bool ignore;

    Detection() {}
    Detection(const QRectF &boundingBox_, float confidence_ = -1, bool ignore_ = false)
        : boundingBox(boundingBox_), confidence(confidence_), ignore(ignore_) {}

    float area() const { return boundingBox.width() * boundingBox.height(); }
    float overlap(const Detection &other) const
    {
        const Detection intersection(boundingBox.intersected(other.boundingBox));
        return intersection.area() / (area() + other.area() - intersection.area());
    }
};

struct SortedDetection
{
    int truth_idx, predicted_idx;
    float overlap;
    SortedDetection() : truth_idx(-1), predicted_idx(-1), overlap(-1) {}
    SortedDetection(int truth_idx_, int predicted_idx_, float overlap_)
        : truth_idx(truth_idx_), predicted_idx(predicted_idx_), overlap(overlap_) {}
    inline bool operator<(const SortedDetection &other) const { return overlap > other.overlap; }
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
    float Recall, FalsePositiveRate, Precision;
    DetectionOperatingPoint() : Recall(-1), FalsePositiveRate(-1), Precision(-1) {}
    DetectionOperatingPoint(float TP, float FP, float totalPositives, float numImages)
        : Recall(TP/totalPositives), FalsePositiveRate(FP/numImages), Precision(TP/(TP+FP)) {}
};

static QStringList computeDetectionResults(const QList<ResolvedDetection> &detections, int totalTrueDetections, int numImages, bool discrete)
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
                if (prevFP / numImages < 0.1 && FP / numImages > 0.1 && discrete) {
                    qDebug("TAR @ FAR => %f : 0.1", TP / totalTrueDetections);
                    qDebug("Confidence: %f", detection.confidence);
                } else if (prevFP / numImages < 0.01 && FP / numImages > 0.01 && discrete) {
                    qDebug("TAR @ FAR => %f : 0.01", TP / totalTrueDetections);
                    qDebug("Confidence: %f", detection.confidence);
                }
                points.append(DetectionOperatingPoint(TP, FP, totalTrueDetections, numImages));
                prevFP = FP;
            }
        }
    }

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
        dets.append(Detection(rect, isTruth ? -1 : f.get<float>("Confidence", -1), f.get<bool>("Ignore", false)));
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
    foreach (const File &f, truth)
        allDetections[f.name].truth.append(getDetections(truthDetectKey, f, true));
    foreach (const File &f, predicted)
        if (allDetections.contains(f.name)) allDetections[f.name].predicted.append(getDetections(predictedDetectKey, f, false));
    return allDetections;
}

static inline int getNumberOfImages(const QMap<QString, Detections> detections)
{   
    return detections.keys().size();
}

static int associateGroundTruthDetections(QList<ResolvedDetection> &resolved, QList<ResolvedDetection> &falseNegative, QMap<QString, Detections> &all, QRectF &offsets)
{
    float dLeftTotal = 0.0, dRightTotal = 0.0, dTopTotal = 0.0, dBottomTotal = 0.0;
    int count = 0, totalTrueDetections = 0;

    foreach (Detections detections, all.values()) {
        totalTrueDetections += detections.truth.size();
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
                Detection newPredicted(QRectF(x, y, width, height), 0.0);

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

            if (!truth.ignore) resolved.append(ResolvedDetection(predicted.confidence, detection.overlap));

            removedTruth.append(detection.truth_idx);
            removedPredicted.append(detection.predicted_idx);

            if (offsets.x() == 0 && detection.overlap > 0.3) {
                count++;
                float width = predicted.boundingBox.width();
                dLeftTotal += (truth.boundingBox.left() - predicted.boundingBox.left()) / width;
                dRightTotal += (truth.boundingBox.right() - predicted.boundingBox.right()) / width;
                dTopTotal += (truth.boundingBox.top() - predicted.boundingBox.top()) / width;
                dBottomTotal += (truth.boundingBox.bottom() - predicted.boundingBox.bottom()) / width;
            }
        }

        for (int i = 0; i < detections.predicted.size(); i++)
            if (!removedPredicted.contains(i)) resolved.append(ResolvedDetection(detections.predicted[i].confidence, 0));
        for (int i = 0; i < detections.truth.size(); i++)
            if (!removedTruth.contains(i) && !detections.truth[i].ignore) falseNegative.append(ResolvedDetection(-std::numeric_limits<float>::max(), 0));
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

float EvalDetection(const QString &predictedGallery, const QString &truthGallery, const QString &csv, bool normalize, int minSize, int maxSize)
{
    qDebug("Evaluating detection of %s against %s", qPrintable(predictedGallery), qPrintable(truthGallery));
    // Organized by file, QMap used to preserve order
    QMap<QString, Detections> allDetections = getDetections(predictedGallery, truthGallery);

    // Remove any bounding boxes with a side smaller than minSize
    if (minSize > 0) {
        qDebug("Removing boxes smaller than %d\n", minSize);
        QMap<QString, Detections> allFilteredDetections;
        foreach (QString key, allDetections.keys()) {
            Detections detections = allDetections[key];
            Detections filteredDetections;
            for (int i = 0; i < detections.predicted.size(); i++) {
                QRectF box = detections.predicted[i].boundingBox;
                if (min(box.width(), box.height()) > sqrt(0.5 * pow(minSize, 2))) {
                    filteredDetections.predicted.append(detections.predicted[i]);
                }
            }
            
            for (int i = 0; i < detections.truth.size(); i++) {
                QRectF box = detections.truth[i].boundingBox;
                if (min(box.width(), box.height()) > minSize) {
                    filteredDetections.truth.append(detections.truth[i]);
                }
            }
            if (!filteredDetections.truth.empty()) allFilteredDetections[key] = filteredDetections;
        }
        allDetections = allFilteredDetections;
    }

    // Remove any bounding boxes with no side smaller than maxSize
    if (maxSize > 0) {
        qDebug("Removing boxes larger than %d\n", maxSize);
        QMap<QString, Detections> allFilteredDetections;
        foreach (QString key, allDetections.keys()) {
            Detections detections = allDetections[key];
            Detections filteredDetections;
            for (int i = 0; i < detections.predicted.size(); i++) {
                QRectF box = detections.predicted[i].boundingBox;
                if (min(box.width(), box.height()) < sqrt(0.5 * pow(maxSize, 2))) {
                    filteredDetections.predicted.append(detections.predicted[i]);
                }
            }

            for (int i = 0; i < detections.truth.size(); i++) {
                QRectF box = detections.truth[i].boundingBox;
                if (min(box.width(), box.height()) < maxSize) {
                    filteredDetections.truth.append(detections.truth[i]);
                }
            }
            if (!filteredDetections.truth.empty()) allFilteredDetections[key] = filteredDetections;
        }
        allDetections = allFilteredDetections;
    }

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
    lines.append(computeDetectionResults(resolvedDetections, totalTrueDetections, getNumberOfImages(allDetections), true));
    lines.append(computeDetectionResults(resolvedDetections, totalTrueDetections, getNumberOfImages(allDetections), false));

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

static void projectAndWrite(Transform *t, const Template &src, const QString &filePath)
{
    Template dst;
    t->project(src,dst);
    OpenCVUtils::saveImage(dst.m(),filePath);
}

float EvalLandmarking(const QString &predictedGallery, const QString &truthGallery, const QString &csv, int normalizationIndexA, int normalizationIndexB, int sampleIndex, int totalExamples)
{
    qDebug("Evaluating landmarking of %s against %s", qPrintable(predictedGallery), qPrintable(truthGallery));
    TemplateList predicted(TemplateList::fromGallery(predictedGallery));
    TemplateList truth(TemplateList::fromGallery(truthGallery));
    QStringList predictedNames = File::get<QString>(predicted, "name");
    QStringList truthNames = File::get<QString>(truth, "name");

    int skipped = 0;
    QList< QList<float> > pointErrors;
    QList<float> imageErrors;
    QList<float> normalizedLengths;
    for (int i=0; i<predicted.size(); i++) {
        const QString &predictedName = predictedNames[i];
        const int truthIndex = truthNames.indexOf(predictedName);
        if (truthIndex == -1) qFatal("Could not identify ground truth for file: %s", qPrintable(predictedName));
        const QList<QPointF> predictedPoints = predicted[i].file.points();
        const QList<QPointF> truthPoints = truth[truthIndex].file.points();
        if (predictedPoints.size() != truthPoints.size() || truthPoints.contains(QPointF(-1,-1))) {
            predicted.removeAt(i);
            predictedNames.removeAt(i);
            truth.removeAt(i);
            truthNames.removeAt(i);
            i--; skipped++;
            continue;
        }

        while (pointErrors.size() < predictedPoints.size())
            pointErrors.append(QList<float>());

        // Want to know error for every image.

        if (normalizationIndexA >= truthPoints.size()) qFatal("Normalization index A is out of range.");
        if (normalizationIndexB >= truthPoints.size()) qFatal("Normalization index B is out of range.");
        const float normalizedLength = QtUtils::euclideanLength(truthPoints[normalizationIndexB] - truthPoints[normalizationIndexA]);
        normalizedLengths.append(normalizedLength);
        float totalError = 0;
        for (int j=0; j<predictedPoints.size(); j++) {
            float error = QtUtils::euclideanLength(predictedPoints[j] - truthPoints[j])/normalizedLength;
            totalError += error;
            pointErrors[j].append(error);
        }
        imageErrors.append(totalError/predictedPoints.size());
    }

    qDebug() << "Skipped" << skipped << "files due to point size mismatch.";

    QList<float> averagePointErrors; averagePointErrors.reserve(pointErrors.size());

    QStringList lines;
    lines.append("Plot,X,Y");

    QtUtils::touchDir(QDir("landmarking_examples_truth"));
    QtUtils::touchDir(QDir("landmarking_examples_predicted"));

    // Example
    {
        QScopedPointer<Transform> t(Transform::make("Open+Draw(verbose,rects=false,location=false)",NULL));

        QString filePath = "landmarking_examples_truth/"+truth[sampleIndex].file.fileName();
        projectAndWrite(t.data(), truth[sampleIndex],filePath);
        lines.append("Sample,"+filePath+","+QString::number(truth[sampleIndex].file.points().size()));
    }

    // Get best and worst performing examples
    QList< QPair<float,int> > exampleIndices = Common::Sort(imageErrors,true);

    QScopedPointer<Transform> t(Transform::make("Open+Draw(rects=false)",NULL));

    for (int i=0; i<totalExamples; i++) {
        QString filePath = "landmarking_examples_truth/"+truth[exampleIndices[i].second].file.fileName();
        projectAndWrite(t.data(), truth[exampleIndices[i].second],filePath);
        lines.append("EXT,"+filePath+","+QString::number(exampleIndices[i].first));

        filePath = "landmarking_examples_predicted/"+predicted[exampleIndices[i].second].file.fileName();
        projectAndWrite(t.data(), predicted[exampleIndices[i].second],filePath);
        lines.append("EXP,"+filePath+","+QString::number(exampleIndices[i].first));
    }

    for (int i=exampleIndices.size()-1; i>exampleIndices.size()-totalExamples-1; i--) {
        QString filePath = "landmarking_examples_truth/"+truth[exampleIndices[i].second].file.fileName();
        projectAndWrite(t.data(), truth[exampleIndices[i].second],filePath);
        lines.append("EXT,"+filePath+","+QString::number(exampleIndices[i].first));

        filePath = "landmarking_examples_predicted/"+predicted[exampleIndices[i].second].file.fileName();
        projectAndWrite(t.data(), predicted[exampleIndices[i].second],filePath);
        lines.append("EXP,"+filePath+","+QString::number(exampleIndices[i].first));
    }

    for (int i=0; i<pointErrors.size(); i++) {
        std::sort(pointErrors[i].begin(), pointErrors[i].end());
        averagePointErrors.append(Common::Mean(pointErrors[i]));
        const QList<float> &pointError = pointErrors[i];
        const int keep = qMin(Max_Points, pointError.size());
        for (int j=0; j<keep; j++)
            lines.append(QString("Box,%1,%2").arg(QString::number(i), QString::number(pointError[j*(pointError.size()-1)/(keep-1)])));
    }

    const float averagePointError = Common::Mean(averagePointErrors);

    lines.append(QString("AvgError,0,%1").arg(averagePointError));
    lines.append(QString("NormLength,0,%1").arg(Common::Mean(normalizedLengths)));

    QtUtils::writeFile(csv, lines);

    qDebug("Average Error for all Points: %.3f", averagePointError);

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

        if (predicted[i].file.contains(predictedProperty) && truth[i].file.contains(truthProperty)) {
            float difference = predicted[i].file.get<float>(predictedProperty) - truth[i].file.get<float>(truthProperty);

            rmsError += pow(difference, 2.f);
            maeError += fabsf(difference);
            truthValues.append(QString::number(truth[i].file.get<float>(truthProperty)));
            predictedValues.append(QString::number(predicted[i].file.get<float>(predictedProperty)));
        }
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

void readKNN(size_t &probeCount, size_t &k, QVector<Candidate> &neighbors, const QString &fileName)
{
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly))
        qFatal("Failed to open k-NN file for reading!");
    file.read((char*) &probeCount, sizeof(size_t));
    file.read((char*) &k, sizeof(size_t));
    neighbors.resize(probeCount * k);

    file.read((char*) neighbors.data(), probeCount * k * sizeof(Candidate));
}

void readKNNTruth(size_t probeCount, QVector< QList<size_t> > &groundTruth, const QString &fileName)
{
    groundTruth.reserve(probeCount);
    QFile truthFile(fileName);
    if (!truthFile.open(QFile::ReadOnly | QFile::Text))
        qFatal("Failed to open k-NN ground truth file for reading!");
    size_t i=0;
    while (!truthFile.atEnd()) {
        const QString line = truthFile.readLine().trimmed();
        if (!line.isEmpty())
            foreach (const QString &index, line.split('\t')) {
                bool ok;
                groundTruth[i].append(index.toLong(&ok));
                if (!ok)
                    qFatal("Failed to parse long in k-NN ground truth!");
            }
        i++;
    }
    if (i != probeCount)
        qFatal("Invalid ground truth file!");
}

void EvalKNN(const QString &knnGraph, const QString &knnTruth, const QString &csv)
{
    qDebug("Evaluating k-NN of %s against %s", qPrintable(knnGraph), qPrintable(knnTruth));

    size_t probeCount;
    size_t k;
    QVector<Candidate> neighbors;
    readKNN(probeCount, k, neighbors, knnGraph);

    /*
     * Read the ground truth from disk.
     * Line i contains the template indicies of the mates for probe i.
     * See the `gtGallery` implementation for details.
     */
    QVector< QList<size_t> > truth(probeCount);
    readKNNTruth(probeCount, truth, knnTruth);

    /*
     * For each probe, record the similarity of the highest mate (if one exists) and the highest non-mate.
     */
    QVector<int> firstGenuineReturns(probeCount, 0);
    QList<float> matedSimilarities, unmatedSimilarities;
    size_t numMatedSearches = 0, numUnmatedSearches = 0;
    for (size_t i=0; i<probeCount; i++) {
        const QList<size_t> &mates = truth[i];
        bool recordedHighestMatedSimilarity = false;
        bool recordedHighestUnmatedSimilarity = false;
        if (!mates.empty()) {
            numMatedSearches++;
            recordedHighestUnmatedSimilarity = true;
        } else {
            numUnmatedSearches++;
            recordedHighestMatedSimilarity = true;
        }

        for (size_t j=0; j<k; j++) {
            const Candidate &neighbor = neighbors[i*k+j];

            if (mates.contains(neighbor.index)) {
                // Found a mate
                if (!recordedHighestMatedSimilarity) {
                    matedSimilarities.append(neighbor.similarity);
                    recordedHighestMatedSimilarity = true;
                }
                if (firstGenuineReturns[i] < 1) firstGenuineReturns[i] = abs(firstGenuineReturns[i])+1;
            } else {
                // Found a non-mate
                if (!recordedHighestUnmatedSimilarity) {
                    unmatedSimilarities.append(neighbor.similarity);
                    recordedHighestUnmatedSimilarity = true;
                }
                if (firstGenuineReturns[i] < 1) firstGenuineReturns[i]--;
            }

            if (recordedHighestMatedSimilarity && recordedHighestUnmatedSimilarity)
                break; // we can stop scanning the candidate list for this probe
        }
    }

    // Sort the similarity scores lowest-to-highest
    std::sort(matedSimilarities.begin(), matedSimilarities.end());
    std::sort(unmatedSimilarities.begin(), unmatedSimilarities.end());
    const size_t numMatedSimilarities = matedSimilarities.size();

    if (numMatedSearches == 0)
        qFatal("No mated searches!");

    if (numUnmatedSearches == 0)
        qFatal("No unmated searches!");


    qDebug("Rank-%d Return Rate: %.3f", 1, getCMC(firstGenuineReturns, 1, numMatedSearches));
    if (k >=5)
        qDebug("Rank-%d Return Rate: %.3f", 5, getCMC(firstGenuineReturns, 5, numMatedSearches));
    if (k >=10)
        qDebug("Rank-%d Return Rate: %.3f", 10, getCMC(firstGenuineReturns, 10, numMatedSearches));

    qDebug("Rank-%zu Return Rate: %.3f", k, double(numMatedSimilarities) / double(numMatedSearches));

    /*
     * Iterate through the similarity scores highest-to-lowest,
     * for each threshold count the number mated and unmated searches,
     * record the corresponding FPIR and FNIR values for the threshold.
     */
    QList<OperatingPoint> operatingPoints;
    size_t matedCount = 0, previousMatedCount = 0;
    size_t unmatedCount = 0, previousUnmatedCount = 0;
    while (!matedSimilarities.empty()) {
        const float threshold = matedSimilarities.back();
        while (!matedSimilarities.empty() && (matedSimilarities.back() >= threshold)) {
            matedSimilarities.removeLast();
            matedCount++;
        }
        while (!unmatedSimilarities.empty() && (unmatedSimilarities.back() >= threshold)) {
            unmatedSimilarities.removeLast();
            unmatedCount++;
        }
        if ((unmatedCount > previousUnmatedCount) && (matedCount > previousMatedCount)) {
            previousMatedCount = matedCount;
            previousUnmatedCount = unmatedCount;
            operatingPoints.append(OperatingPoint(threshold,
                                                  double(unmatedCount) / double(numUnmatedSearches),
                                                  1.0 - double(matedCount) / double(numMatedSearches)));
        }
    }

    if (!csv.isEmpty()) {
        // Open the output file
        QFile ietFile(csv);
        if (!ietFile.open(QFile::WriteOnly | QFile::Text))
            qFatal("Failed to open IET file for writing!");
        ietFile.write("Plot,X,Y,Z\n");
        // Write CMC
        const int Max_Retrieval = min(200, (int)k);
        for (int i=1; i<=Max_Retrieval; i++) {
            const float retrievalRate = getCMC(firstGenuineReturns, i, numMatedSearches);
            ietFile.write(qPrintable(QString("CMC,%1,%2,0\n").arg(QString::number(i), QString::number(retrievalRate))));
        }

        foreach(const OperatingPoint &operatingPoint, operatingPoints)
            ietFile.write(qPrintable("IET," +
                                     QString::number(operatingPoint.FAR) + "," +
                                     QString::number(operatingPoint.TAR) + "," +
                                     QString::number(operatingPoint.score) + "\n"));
    }

    qDebug("FNIR @ FPIR = 0.1:   %.3f", 1-getOperatingPointGivenFAR(operatingPoints, 0.1).TAR);
    qDebug("FNIR @ FPIR = 0.01:  %.3f", 1-getOperatingPointGivenFAR(operatingPoints, 0.01).TAR);
}

} // namespace br
