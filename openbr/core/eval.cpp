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
#include "openbr/core/evalutils.h"
#include <QMapIterator>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace EvalUtils;

namespace br
{

static const int Max_Points = 1000; // Maximum number of points to render on plots

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

static OperatingPoint getOperatingPoint(const QList<OperatingPoint> &operatingPoints, const QString key, const float value)
{
    bool invert      = (key == "Score") && (operatingPoints.first().score > operatingPoints.last().score);
    int index        = invert ? operatingPoints.size()-1 : 0;
    const int break_ = invert ? 0 : operatingPoints.size();
    while ((key == "Score" ? operatingPoints[index].score :
            key == "FAR"   ? operatingPoints[index].FAR   :
                             operatingPoints[index].TAR) < value) {
        index = index + (invert ? -1 : 1);
        if (index == break_) {
            if (key == "Score")
                return OperatingPoint(value, (invert ? operatingPoints.first() : operatingPoints.last()).FAR, (invert ? operatingPoints.first() : operatingPoints.last()).TAR);
            else if (key == "FAR")
                return OperatingPoint(operatingPoints.last().score, value, operatingPoints.last().TAR);
            return OperatingPoint(operatingPoints.last().score, operatingPoints.last().FAR, value);
        }
    }

    if (
        (key == "Score" && operatingPoints[index].score == value)
        || (key == "FAR" && operatingPoints[index].FAR == value)
        || (key == "TAR" && operatingPoints[index].TAR == value)
        ) {
        // If the OperatingPoint is explicitly found
        return operatingPoints[index];
    }

    const int index2 = (key == "Score" ? std::min(index+1, operatingPoints.size()-1) : index-1);
    const float FAR1   = (index == 0 ? 0 : operatingPoints[index2].FAR);
    const float TAR1   = (index == 0 ? 0 : operatingPoints[index2].TAR);
    const float score1 = (index == 0 ? operatingPoints[index].score : operatingPoints[index2].score);
    const float FAR2   = operatingPoints[index].FAR;
    const float TAR2   = operatingPoints[index].TAR;
    const float score2 = operatingPoints[index].score;

    const float denFAR = (FAR1 == FAR2 ? 1e-10 : (FAR2 - FAR1));
    const float denScore = (score1 == score2 ? 1e-10 : (score2 - score1));

    const float mFAR   = (FAR2 - FAR1) / denScore;
    const float bFAR   = FAR1 - mFAR*score1;
    const float mTAR   = (TAR2 - TAR1) / (key == "Score" ? denScore : denFAR);
    const float bTAR   = TAR1 - mTAR*(key == "Score" ? score1 : FAR1);
    const float mScore = (score2 - score1) / denFAR;
    const float bScore = score1 - mScore*FAR1;

    if (key == "Score")
        return OperatingPoint(value, mFAR*value + bFAR, mTAR*value + bTAR);
    else if (key == "FAR")
        return OperatingPoint(mScore * value + bScore, value, mTAR * value + bTAR);
    return OperatingPoint(mScore * ((value - bTAR) / mTAR) + bScore, (value - bTAR) / mTAR, value);
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
    std::vector<Comparison> comparisons; comparisons.reserve(simmat.rows*simmat.cols);

    // Flags rows as being mated or non-mated searches
    // Positive value: mated search, negative value: non-mated search
    // Value of 0: ignored search
    std::vector<int> genuineSearches(simmat.rows, 0);

    size_t totalGenuineSearches = 0, totalImpostorSearches = 0;
    size_t genuineCount = 0, impostorCount = 0, numNaNs = 0;
    for (int i=0; i<simmat.rows; i++) {
        for (int j=0; j<simmat.cols; j++) {
            const BEE::MaskValue mask_val = mask.at<BEE::MaskValue>(i,j);
            const BEE::SimmatValue simmat_val = simmat.at<BEE::SimmatValue>(i,j);
            if (mask_val == BEE::DontCare) continue;
            if (simmat_val != simmat_val) { numNaNs++; continue; }
            Comparison comparison(simmat_val, j, i, mask_val == BEE::Match);
            comparisons.push_back(comparison);
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

    if (numNaNs > 0) qWarning("Encountered %ld NaN scores!", numNaNs);
    if (genuineCount == 0) qFatal("No genuine scores!");
    if (impostorCount == 0) qFatal("No impostor scores!");

    // Sort comparisons by simmat_val (score)
    std::sort(comparisons.begin(), comparisons.end());

    QList<OperatingPoint> operatingPoints;
    QList<OperatingPoint> searchOperatingPoints;
    std::vector<float> genuines; genuines.reserve(sqrt((float)comparisons.size()));
    std::vector<float> impostors; impostors.reserve(comparisons.size());
    QVector<int> firstGenuineReturns(simmat.rows, 0);

    size_t falsePositives = 0, previousFalsePositives = 0;
    size_t truePositives = 0, previousTruePositives = 0;
    size_t falseSearches = 0, previousFalseSearches = 0;
    size_t trueSearches = 0, previousTrueSearches = 0;
    size_t index = 0;
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
                genuines.push_back(comparison.score);
                if (firstGenuineReturns[comparison.query] < 1)
                    firstGenuineReturns[comparison.query] = (comparison.score == -std::numeric_limits<float>::max())
                                                          ? std::numeric_limits<int>::max()
                                                          : abs(firstGenuineReturns[comparison.query]) + 1;
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
                impostors.push_back(comparison.score);
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
        for (size_t i = EERIndex+1; i < comparisons.size(); i++) {
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

        OperatingPoint operatingPointFAR = getOperatingPoint(operatingPoints, "FAR", FAR);
        OperatingPoint operatingPointTAR = getOperatingPoint(operatingPoints, "TAR", 1-FRR);
        OperatingPoint searchOperatingPoint = getOperatingPoint(searchOperatingPoints, "FAR", FPIR);
        lines.append(QString("DET,%1,%2").arg(QString::number(FAR),
                                              QString::number(1-operatingPointFAR.TAR)));
        lines.append(QString("FAR,%1,%2").arg(QString::number(operatingPointFAR.score),
                                              QString::number(FAR)));
        lines.append(QString("FRR,%1,%2").arg(QString::number(operatingPointTAR.score),
                                              QString::number(FRR)));
        lines.append(QString("IET,%1,%2").arg(QString::number(searchOperatingPoint.FAR),
                                              QString::number(1-searchOperatingPoint.TAR)));
    }

    // Write FRR@FAR Table (FF)
    foreach (float FAR, QList<float>() << 1e-6 << 1e-5 << 1e-4 << 1e-3 << 1e-2 << 1e-1)
        lines.append(qPrintable(QString("FF,%1,%2").arg(
                                QString::number(FAR, 'f'),
                                QString::number(1-getOperatingPoint(operatingPoints, "FAR", FAR).TAR, 'f', 6))));

    // Write FAR@TAR Table (FT)
    foreach (float TAR, QList<float>() << 0.4 << 0.5 << 0.65 << 0.75 << 0.85 << 0.95)
        lines.append(qPrintable(QString("FT,%1,%2").arg(
                                QString::number(TAR, 'f', 2),
                                QString::number(getOperatingPoint(operatingPoints, "TAR", TAR).FAR, 'f', 3))));

    // Write FAR@Score Table (SF) and TAR@Score table (ST)
    foreach(const float score, QList<float>() << 0.05 << 0.1 << 0.15 << 0.2 << 0.25 << 0.3 << 0.35 << 0.4 << 0.45 << 0.5
                                              << 0.55 << 0.6 << 0.65 << 0.7 << 0.75 << 0.8 << 0.85 << 0.9 << 0.95) {
        const OperatingPoint op = getOperatingPoint(operatingPoints, "Score", score);
        lines.append(qPrintable(QString("SF,%1,%2").arg(
                                QString::number(score, 'f', 2),
                                QString::number(op.FAR))));
        lines.append(qPrintable(QString("ST,%1,%2").arg(
                                QString::number(score, 'f', 2),
                                QString::number(op.TAR))));
    }

    // Write CMC Table (CT)
    lines.append(qPrintable(QString("CT,1,%1").arg(QString::number(getCMC(firstGenuineReturns, 1), 'f', 3))));
    lines.append(qPrintable(QString("CT,5,%1").arg(QString::number(getCMC(firstGenuineReturns, 5), 'f', 3))));
    lines.append(qPrintable(QString("CT,10,%1").arg(QString::number(getCMC(firstGenuineReturns, 10), 'f', 3))));
    lines.append(qPrintable(QString("CT,20,%1").arg(QString::number(getCMC(firstGenuineReturns, 20), 'f', 3))));
    lines.append(qPrintable(QString("CT,50,%1").arg(QString::number(getCMC(firstGenuineReturns, 50), 'f', 3))));
    lines.append(qPrintable(QString("CT,100,%1").arg(QString::number(getCMC(firstGenuineReturns, 100), 'f', 3))));

    // Write FAR/TAR Bar Chart (BC)
    lines.append(qPrintable(QString("BC,0.0001,%1").arg(QString::number(getOperatingPoint(operatingPoints, "FAR", 0.0001).TAR, 'f', 3))));
    lines.append(qPrintable(QString("BC,0.001,%1").arg(QString::number(result = getOperatingPoint(operatingPoints, "FAR", 0.001).TAR, 'f', 3))));

    // Attempt to read template size from enrolled gallery and write to output CSV
    size_t maxSize(0);
    if (target.endsWith(".gal") && QFileInfo(target).exists()) {
        foreach (const Template &t, TemplateList::fromGallery(target)) maxSize = max(maxSize, t.bytes());
        lines.append(QString("TS,,%1").arg(QString::number(maxSize)));
    }

    // Write SD
    int points = qMin(qMin((size_t)Max_Points, genuines.size()), impostors.size());
    if (points > 1) {
        for (int i=0; i<points; i++) {
            float genuineScore = genuines[double(i) / double(points-1) * double(genuines.size()-1)];
            float impostorScore = impostors[double(i) / double(points-1) * double(impostors.size()-1)];
            if (genuineScore == -std::numeric_limits<float>::max()) genuineScore = minGenuineScore;
            if (impostorScore == -std::numeric_limits<float>::max()) impostorScore = minImpostorScore;
            lines.append(QString("SD,%1,Genuine").arg(QString::number(genuineScore)));
            lines.append(QString("SD,%1,Impostor").arg(QString::number(impostorScore)));
        }
    }

    // Write Cumulative Match Characteristic (CMC) curve
    const int Max_Retrieval = 200;
    const QList<int> Report_Retrieval_List = QList<int>() << 1 << 5 << 10 << 20 << 50 << 100;
    for (int i=1; i<=Max_Retrieval; i++) {
        const float retrievalRate = getCMC(firstGenuineReturns, i);
        lines.append(qPrintable(QString("CMC,%1,%2").arg(QString::number(i), QString::number(retrievalRate))));
    }

    QtUtils::writeFile(csv, lines);
    if (maxSize > 0) qDebug("Template Size: %i bytes", (int)maxSize);
    foreach (float FAR, QList<float>() << 1e-1 << 1e-2 << 1e-3 << 1e-4 << 1e-5 << 1e-6) {
        const OperatingPoint op = getOperatingPoint(operatingPoints, "FAR", FAR);
        printf("TAR & Similarity @ FAR = %.0e: %.4f %.3f\n", FAR, op.TAR, op.score);
    }
    printf("\n");
    foreach (float FPIR, QList<float>() << 0.1 << 0.01) {
        const OperatingPoint op = getOperatingPoint(searchOperatingPoints, "FAR", FPIR);
        printf("FNIR @ FPIR = %.0e: %.3f\n", FPIR, 1-op.TAR);
    }
    printf("\n");
    foreach (const int Report_Retrieval, Report_Retrieval_List)
        printf("Retrieval Rate @ Rank = %d: %.3f\n", Report_Retrieval, getCMC(firstGenuineReturns, Report_Retrieval));

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

void openAndReadMatrixHeader(QFile &file, size_t &rows, size_t &cols)
{
    bool success = file.open(QFile::ReadOnly);
    if (!success) qFatal("Unable to open %s for reading.", qPrintable(file.fileName()));

    // Ignore these lines
    file.readLine();
    file.readLine();
    file.readLine();

    // Get matrix size
    const QStringList words = QString(file.readLine()).split(" ");
    rows = words[1].toLong();
    cols = words[2].toLong();
}

// Helper struct for statistics accumulation
struct Counter
{
    double truePositive, falsePositive, falseNegative;
    Counter() : truePositive(0), falsePositive(0), falseNegative(0) {}
};

float InplaceEval(const QString &simmat, const QString &mask, const QString &csv)
{
    QFile simmatFile(simmat), maskFile(mask);
    size_t simmatRows, simmatCols, maskRows, maskCols;
    openAndReadMatrixHeader(simmatFile, simmatRows, simmatCols);
    openAndReadMatrixHeader(maskFile, maskRows, maskCols);

    if ((simmatRows != maskRows) || (simmatCols != maskCols))
        qFatal("Simmat and Mask size mismatch!");

    // DET stats
    size_t genuineCount(0), impostorCount(0), numNaNs(0);
    QHash<float, Counter> stats;

    // IET stats
    size_t genuineSearchCount(0), impostorSearchCount(0);
    QHash<float, Counter> ietStats;

    // CMC stats
    QVector<int> firstGenuineReturns(simmatRows, 0);

    Mat simmatRow, maskRow;
    simmatRow.create(1, simmatCols, CV_32FC1);
    maskRow.create(1, maskCols, CV_8UC1);
    for (size_t i=0; i<simmatRows; i++) {
        qint64 bytesRead = simmatFile.read((char*)simmatRow.data, simmatCols * sizeof(BEE::SimmatValue));
        if (bytesRead != simmatCols * sizeof(BEE::SimmatValue))
            qFatal("Didn't read complete matrix row!");

        bytesRead = maskFile.read((char*)maskRow.data, maskCols * sizeof(BEE::MaskValue));
        if (bytesRead != maskCols * sizeof(BEE::MaskValue))
            qFatal("Didn't read complete mask row!");

        // DET
        int impostorsAboveGenuine(0);
        bool hasGenuine(false), hasImpostor(false);
        float highestImpostor(-std::numeric_limits<float>::max()), highestGenuine(-std::numeric_limits<float>::max());
        for (size_t j=0; j<simmatCols; j++) {
            const BEE::SimmatValue sim_val = simmatRow.at<BEE::SimmatValue>(0, j);
            const BEE::MaskValue mask_val = maskRow.at<BEE::MaskValue>(0, j);
            if (mask_val == BEE::DontCare) continue;
            if (sim_val != sim_val) { numNaNs++; continue; }

            const bool genuine = (mask_val == BEE::Match);
            if (genuine) {
                hasGenuine = true;
                genuineCount++;
                stats[sim_val].truePositive++;
                if (sim_val > highestImpostor)
                    impostorsAboveGenuine = 0;
                if (sim_val > highestGenuine)
                    highestGenuine = sim_val;
            } else {
                hasImpostor = true;
                impostorCount++;
                stats[sim_val].falsePositive++;
                if (sim_val >= highestGenuine)
                    impostorsAboveGenuine++;
                if (sim_val > highestImpostor)
                    highestImpostor = sim_val;
            }
        }

        // CMC
        firstGenuineReturns[i] = hasGenuine ? impostorsAboveGenuine + 1 : -impostorsAboveGenuine;

        // IET
        if (hasGenuine) { // true search
            genuineSearchCount++;
            ietStats[highestGenuine].truePositive++;
        } else if (hasImpostor) { // false search
            impostorSearchCount++;
            ietStats[highestImpostor].falsePositive++;
        }
    }

    if (numNaNs > 0) qWarning("Encountered %ld NaN scores!", numNaNs);
    if (genuineCount == 0) qFatal("No genuine scores!");
    if (impostorCount == 0) qFatal("No impostor scores!");

    // Write Metadata table
    QStringList lines;
    lines.append("Plot,X,Y");
    lines.append("Metadata,"+QString::number(simmatCols)+",Gallery");
    lines.append("Metadata,"+QString::number(simmatRows)+",Probe");
    lines.append("Metadata,"+QString::number(genuineCount)+",Genuine");
    lines.append("Metadata,"+QString::number(impostorCount)+",Impostor");
    lines.append("Metadata,"+QString::number(simmatCols*simmatRows-(genuineCount+impostorCount))+",Ignored");

    // SD points
    size_t sdPoints = qMin(qMin((size_t)Max_Points, genuineCount), impostorCount);
    size_t genuineRate = genuineCount / sdPoints;
    size_t impostorRate = impostorCount / sdPoints;
    size_t genuinesSampled(0), sampleNextGenuine(0), impostorsSampled(0), sampleNextImpostor(0);

    // DET and SD sampling
    size_t fps(0), previous_fps(0), tps(0), previous_tps(0);
    QList<float> thresholds = stats.keys();
    std::sort(thresholds.begin(), thresholds.end());
    std::reverse(thresholds.begin(), thresholds.end());
    QList<OperatingPoint> operatingPoints;
    for (int i=0; i<thresholds.size(); i++) {
        const Counter counter = stats[thresholds[i]];
        fps += counter.falsePositive;
        tps += counter.truePositive;
        if ((fps > previous_fps) && tps > previous_tps) {
            operatingPoints.append(OperatingPoint(thresholds[i], double(fps)/impostorCount, double(tps)/genuineCount));
            previous_fps = fps;
            previous_tps = tps;
        }

        // Genuine score sampling
        if ((genuinesSampled < sdPoints) && (counter.truePositive > 0)) {
            for (int sample=0; sample<counter.truePositive; sample++) {
                sampleNextGenuine++;
                if (sampleNextGenuine == genuineRate) {
                    sampleNextGenuine = 0;
                    lines.append(QString("SD,%1,Genuine").arg(QString::number(thresholds[i])));
                    genuinesSampled++;
                    if (genuinesSampled >= sdPoints)
                        break;
                }
            }
        }

        // Impostor score sampling
        if ((impostorsSampled < sdPoints) && (counter.falsePositive > 0)) {
            for (int sample=0; sample<counter.falsePositive; sample++) {
                sampleNextImpostor++;
                if (sampleNextImpostor == impostorRate) {
                    sampleNextImpostor = 0;
                    lines.append(QString("SD,%1,Impostor").arg(QString::number(thresholds[i])));
                    impostorsSampled++;
                    if (impostorsSampled >= sdPoints)
                        break;
                }
            }
        }
    }
    if (operatingPoints.size() == 0) operatingPoints.append(OperatingPoint(1, 1, 1));
    if (operatingPoints.size() == 1) operatingPoints.prepend(OperatingPoint(0, 0, 0));
    if (operatingPoints.size() > 2)  operatingPoints.takeLast(); // Remove point (1,1)

    // IET
    QList<OperatingPoint> searchOperatingPoints;
    if (genuineSearchCount > 0 && impostorSearchCount > 0) {
        fps = 0; tps = 0;
        QList<float> thresholds = ietStats.keys();
        std::sort(thresholds.begin(), thresholds.end());
        std::reverse(thresholds.begin(), thresholds.end());
        for (int i=0; i<thresholds.size(); i++) {
            const Counter counter = ietStats[thresholds[i]];
            fps += counter.falsePositive;
            tps += counter.truePositive;
            searchOperatingPoints.append(OperatingPoint(thresholds[i], double(fps)/impostorSearchCount, double(tps)/genuineSearchCount));
        }
    }
    if (searchOperatingPoints.size() == 0) searchOperatingPoints.append(OperatingPoint(1, 1, 1));
    if (searchOperatingPoints.size() == 1) searchOperatingPoints.prepend(OperatingPoint(0, 0, 0));
    if (searchOperatingPoints.size() > 2)  searchOperatingPoints.takeLast(); // Remove point (1,1)


    // Write Detection Error Tradeoff (DET), PRE, REC, Identification Error Tradeoff (IET)
    float expFAR = std::max(ceil(log10(impostorCount)), 1.0);
    float expFRR = std::max(ceil(log10(genuineCount)), 1.0);
    float expFPIR = std::max(ceil(log10(impostorSearchCount)), 1.0);

    float FARstep = expFAR / (float)(Max_Points - 1);
    float FRRstep = expFRR / (float)(Max_Points - 1);
    float FPIRstep = expFPIR / (float)(Max_Points - 1);

    for (int i=0; i<Max_Points; i++) {
        float FAR = pow(10, -expFAR + i*FARstep);
        float FRR = pow(10, -expFRR + i*FRRstep);
        float FPIR = pow(10, -expFPIR + i*FPIRstep);

        OperatingPoint operatingPointFAR = getOperatingPoint(operatingPoints, "FAR", FAR);
        OperatingPoint operatingPointTAR = getOperatingPoint(operatingPoints, "TAR", 1-FRR);
        OperatingPoint searchOperatingPoint = getOperatingPoint(searchOperatingPoints, "FAR", FPIR);
        lines.append(QString("DET,%1,%2").arg(QString::number(FAR),
                                              QString::number(1-operatingPointFAR.TAR)));
        lines.append(QString("FAR,%1,%2").arg(QString::number(operatingPointFAR.score),
                                              QString::number(FAR)));
        lines.append(QString("FRR,%1,%2").arg(QString::number(operatingPointTAR.score),
                                              QString::number(FRR)));
        lines.append(QString("IET,%1,%2").arg(QString::number(searchOperatingPoint.FAR),
                                              QString::number(1-searchOperatingPoint.TAR)));
    }

    // Write Cumulative Match Characteristic (CMC) curve
    const int Max_Retrieval = 200;
    for (int i=1; i<=Max_Retrieval; i++) {
        const float retrievalRate = getCMC(firstGenuineReturns, i);
        lines.append(qPrintable(QString("CMC,%1,%2").arg(QString::number(i), QString::number(retrievalRate))));
    }

    // Write FRR@FAR Table (FF)
    foreach (float FAR, QList<float>() << 1e-6 << 1e-5 << 1e-4 << 1e-3 << 1e-2 << 1e-1)
      lines.append(qPrintable(QString("FF,%1,%2").arg(
                              QString::number(FAR, 'f'),
                              QString::number(1-getOperatingPoint(operatingPoints, "FAR", FAR).TAR, 'f', 6))));

    // Write FAR@TAR Table (FT)
    foreach (float TAR, QList<float>() << 0.4 << 0.5 << 0.65 << 0.75 << 0.85 << 0.95)
      lines.append(qPrintable(QString("FT,%1,%2").arg(
                         QString::number(TAR, 'f', 2),
                         QString::number(getOperatingPoint(operatingPoints, "TAR", TAR).FAR, 'f', 3))));

    // Write FAR@Score Table (SF) and TAR@Score table (ST)
    foreach(const float score, QList<float>() << 0.05 << 0.1 << 0.15 << 0.2 << 0.25 << 0.3 << 0.35 << 0.4 << 0.45 << 0.5
                                              << 0.55 << 0.6 << 0.65 << 0.7 << 0.75 << 0.8 << 0.85 << 0.9 << 0.95) {
        const OperatingPoint op = getOperatingPoint(operatingPoints, "Score", score);
        lines.append(qPrintable(QString("SF,%1,%2").arg(
                                QString::number(score, 'f', 2),
                                QString::number(op.FAR))));
        lines.append(qPrintable(QString("ST,%1,%2").arg(
                                QString::number(score, 'f', 2),
                                QString::number(op.TAR))));
    }

    // Write FAR/TAR Bar Chart (BC)
    float result = 0;
    lines.append(qPrintable(QString("BC,0.0001,%1").arg(QString::number(getOperatingPoint(operatingPoints, "FAR", 0.0001).TAR, 'f', 3))));
    lines.append(qPrintable(QString("BC,0.001,%1").arg(QString::number(result = getOperatingPoint(operatingPoints, "FAR", 0.001).TAR, 'f', 3))));

    QtUtils::writeFile(csv, lines);

    foreach (float FAR, QList<float>() << 1e-1 << 1e-2 << 1e-3 << 1e-4 << 1e-5 << 1e-6) {
        const OperatingPoint op = getOperatingPoint(operatingPoints, "FAR", FAR);
        printf("TAR & Similarity @ FAR = %.0e: %.4f %.3f\n", FAR, op.TAR, op.score);
    }
    printf("\n");
    foreach (float FPIR, QList<float>() << 0.1 << 0.01) {
        const OperatingPoint op = getOperatingPoint(searchOperatingPoints, "FAR", FPIR);
        printf("FNIR @ FPIR = %.0e: %.3f\n", FPIR, 1-op.TAR);
    }
    printf("\n");
    foreach (const int Report_Retrieval, QList<int>() << 1 << 5 << 10 << 20 << 50 << 100)
        printf("Retrieval Rate @ Rank = %d: %.3f\n", Report_Retrieval, getCMC(firstGenuineReturns, Report_Retrieval));

    return result;
}

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

float EvalDetection(const QString &predictedGallery, const QString &truthGallery, const QString &csv, bool normalize, int minSize, int maxSize, float relativeMinSize, const QString &label, const float truePositiveThreshold)
{
    qDebug("Evaluating detection of %s against %s", qPrintable(predictedGallery), qPrintable(truthGallery));
    // Organized by file, QMap used to preserve order
    QMap<QString, Detections> allDetections = getDetections(predictedGallery, truthGallery);

    // Remove any bounding boxes with a side smaller than minSize
    if (minSize > 0 || relativeMinSize > 0) {
        if (Globals->verbose)
            qDebug("Removing boxes smaller than %d\n", minSize);
        allDetections = filterDetections(allDetections, minSize, true, relativeMinSize);
    }

    // Remove any bounding boxes with no side smaller than maxSize
    if (maxSize > 0) {
        if (Globals->verbose)
            qDebug("Removing boxes larger than %d\n", maxSize);
        allDetections = filterDetections(allDetections, maxSize, false);
    }

    // Optionally, keep only detections with a specific label
    if (!label.isEmpty()) {
        if (Globals->verbose)
            qDebug("Removing detections without label %s\n", qPrintable(label));
        allDetections = filterLabels(allDetections, label);
        if (allDetections.isEmpty())
            qFatal("No detections left after filtering on label. Check your filter");
    }

    QList<ResolvedDetection> resolvedDetections, falseNegativeDetections;
    QRectF normalizations(0, 0, 0, 0);

    // Associate predictions to ground truth
    int totalTrueDetections = associateGroundTruthDetections(resolvedDetections, falseNegativeDetections, allDetections, normalizations, truePositiveThreshold);

    // Redo association of ground truth to predictions with boundingBoxes
    // resized based on the average differences on each side.
    if (normalize) {
        if (Globals->verbose) {
            qDebug("dX = %.3f", normalizations.x());
            qDebug("dY = %.3f", normalizations.y());
            qDebug("dWidth = %.3f", normalizations.width());
            qDebug("dHeight = %.3f", normalizations.height());
        }
        resolvedDetections.clear();
        falseNegativeDetections.clear();
        totalTrueDetections = associateGroundTruthDetections(resolvedDetections, falseNegativeDetections, allDetections, normalizations, truePositiveThreshold);
    }

    if (Globals->verbose) {
        qDebug() << "Total False Negatives: " << falseNegativeDetections.size();
        for (int i=0; i<falseNegativeDetections.size(); i++) {
            Mat img = imread(qPrintable(Globals->path + "/" + falseNegativeDetections[i].filePath));
            qDebug() << falseNegativeDetections[i];
            const Scalar color(0,255,0);
            rectangle(img, OpenCVUtils::toRect(falseNegativeDetections[i].boundingBox), color, 1);
            QtUtils::touchDir(QDir("./falseNegs"));
            imwrite(qPrintable(QString("./falseNegs/%1_%2.jpg").arg(falseNegativeDetections[i].filePath.split("/").last(), QString::number(i))), img);
        }
    }

    std::sort(resolvedDetections.begin(), resolvedDetections.end());
    QStringList lines;
    lines.append("Plot, X, Y");
    QList<DetectionOperatingPoint> points;
    lines.append(computeDetectionResults(resolvedDetections, totalTrueDetections, getNumberOfImages(allDetections), true, points, truePositiveThreshold));
    points.clear();
    lines.append(computeDetectionResults(resolvedDetections, totalTrueDetections, getNumberOfImages(allDetections), false, points, truePositiveThreshold));

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
    qDebug("Average Overlap = %.4f\n", averageOverlap);
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
    QList< QList<float> > pointErrorMagnitudes, pointErrorOrientations;
    QList<float> imageErrors;
    QList<float> normalizedLengths;
    for (int i=0; i<predicted.size(); i++) {
        const QString &predictedName = predictedNames[i];
        int truthIndex;
        if ((i < truthNames.size()) && (truthNames[i] == predictedNames[i]))
            truthIndex = i;
        else
            truthIndex = truthNames.indexOf(predictedName);
        if (truthIndex == -1) qFatal("Could not identify ground truth for file: %s", qPrintable(predictedName));

        const QList<QPointF> predictedPoints = predicted[i].file.points();
        QList<QPointF> truthPoints = truth[truthIndex].file.points();

        // Standardize how we represent unlabeled points here
        const QPointF findNegOne(-1,-1);
        const QPointF findZero(0,0);
        const QPointF replace(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
        for (int j=0; j<truthPoints.size(); j++)
            if ((truthPoints[j] == findNegOne) || (truthPoints[j] == findZero))
                truthPoints[j] = replace;

        if (normalizationIndexA >= truthPoints.size()) qFatal("Normalization index A is out of range.");
        if (normalizationIndexB >= truthPoints.size()) qFatal("Normalization index B is out of range.");
        const float normalizedLength = QtUtils::euclideanLength(truthPoints[normalizationIndexB] - truthPoints[normalizationIndexA]);
        const float normalizedOrientation = QtUtils::orientation(truthPoints[normalizationIndexB], truthPoints[normalizationIndexA]);

        if (// If the landmarks don't match up
            (predictedPoints.size() != truthPoints.size())
            // Or the landmarks used for normalization are missing
            || qIsNaN(normalizedLength)
            // Or the ground truth seems to be for another object in the image
            || (QtUtils::euclideanLength(predictedPoints[normalizationIndexA] - truthPoints[normalizationIndexA]) / normalizedLength >= 0.5)
            || (QtUtils::euclideanLength(predictedPoints[normalizationIndexB] - truthPoints[normalizationIndexB]) / normalizedLength >= 0.5)
            // Or the predicted image FTE'd
            || predicted[i].file.fte || predicted[i].file.getBool("FTE")
           ) {
            predicted.removeAt(i);
            predictedNames.removeAt(i);
            truth.removeAt(i);
            truthNames.removeAt(i);
            i--; skipped++;
            continue;
        }

        while (pointErrorMagnitudes.size() < predictedPoints.size()) {
            pointErrorMagnitudes.append(QList<float>());
            pointErrorOrientations.append(QList<float>());
        }

        // Want to know error for every image.
        normalizedLengths.append(normalizedLength);
        float totalError = 0;
        int totalCount = 0;
        for (int j=0; j<predictedPoints.size(); j++) {
            const float error = QtUtils::euclideanLength(predictedPoints[j] - truthPoints[j])/normalizedLength;
            if (!qIsNaN(error)) {
                totalError += error;
                pointErrorMagnitudes[j].append(error);
                pointErrorOrientations[j].append(QtUtils::orientation(predictedPoints[j], truthPoints[j]) - normalizedOrientation);
                totalCount++;
            }
        }
        imageErrors.append(totalError/totalCount);
    }

    qDebug("Files skipped: %d", skipped);

    // Adjust the point error to not penalize for systematic biases...
    // ... by first calculating the average bias for each point
    QList<QPointF> averagePointBiases;
    for (int i=0; i<pointErrorMagnitudes.size(); i++) {
        const QList<float> &magnitudes   = pointErrorMagnitudes[i];
        const QList<float> &orientations = pointErrorOrientations[i];
        QPointF cumulativePointBias;
        for (int j=0; j<magnitudes.size(); j++) {
            const float m = magnitudes[j];
            const float o = orientations[j];
            cumulativePointBias += QPointF(m*cos(o), m*sin(o));
        }
        averagePointBiases.append(cumulativePointBias / magnitudes.size());
    }

    // ... and then subtracting the average bias from each individual error.
    for (int i=0; i<pointErrorMagnitudes.size(); i++) {
        QList<float> &magnitudes   = pointErrorMagnitudes[i];
        QList<float> &orientations = pointErrorOrientations[i];
        const QPointF &bias = averagePointBiases[i];
        for (int j=0; j<magnitudes.size(); j++) {
            float &m = magnitudes[j];
            float &o = orientations[j];
            QPointF error(m*cos(o), m*sin(o));
            error -= bias;
            // At this point if we added up all the `error` vectors for a
            // landmark they would sum to zero. Josh confirmed this when
            // implementing the bias normalization correction, but removed it
            // from the final implementation.

            // Update the error magnitude for reporting MAE
             m = QtUtils::euclideanLength(error);

            // We don't need to update orientation because we don't use it
            // again, but we do so anyway in the interest of pedantic
            // correctness.
            o = QtUtils::orientation(QPointF(0.f,0.f), error);
        }
    }

    QList<float> averagePointErrors; averagePointErrors.reserve(pointErrorMagnitudes.size());

    QStringList lines;
    lines.append("Plot,X,Y");

    QtUtils::touchDir(QDir("landmarking_examples_truth"));
    QtUtils::touchDir(QDir("landmarking_examples_predicted"));

    // Example
    {
        QScopedPointer<Transform> t(Transform::make("Open+Draw(verbose,rects=false,named=false,location=false)",NULL));

        QString filePath = "landmarking_examples_truth/sample.jpg";
        projectAndWrite(t.data(), truth[sampleIndex],filePath);
        lines.append("Sample,"+filePath+","+QString::number(truth[sampleIndex].file.points().size()));
    }

    // Get best and worst performing examples
    QList< QPair<float,int> > exampleIndices = Common::Sort(imageErrors,true);

    QScopedPointer<Transform> t(Transform::make("Open+CropFromLandmarks(paddingHorizontal=.3,paddingVertical=.3,shiftPoints=true)+Resize(128,method=Area)+Draw(rects=false,pointRadius=2)",NULL));

    for (int i=0; i<totalExamples; i++) {
        QString filePath = "landmarking_examples_truth/"+truth[exampleIndices[i].second].file.fileName();
        projectAndWrite(t.data(), truth[exampleIndices[i].second],filePath);
        lines.append("EXT,"+filePath+":"+truth[exampleIndices[i].second].file.name+","+QString::number(exampleIndices[i].first));

        filePath = "landmarking_examples_predicted/"+predicted[exampleIndices[i].second].file.fileName();
        projectAndWrite(t.data(), predicted[exampleIndices[i].second],filePath);
        lines.append("EXP,"+filePath+":"+ predicted[exampleIndices[i].second].file.name+","+QString::number(exampleIndices[i].first));
    }

    for (int i=exampleIndices.size()-1; i>exampleIndices.size()-totalExamples-1; i--) {
        QString filePath = "landmarking_examples_truth/"+truth[exampleIndices[i].second].file.fileName();
        projectAndWrite(t.data(), truth[exampleIndices[i].second],filePath);
        lines.append("EXT,"+filePath+":"+truth[exampleIndices[i].second].file.name+","+QString::number(exampleIndices[i].first));

        filePath = "landmarking_examples_predicted/"+predicted[exampleIndices[i].second].file.fileName();
        projectAndWrite(t.data(), predicted[exampleIndices[i].second],filePath);
        lines.append("EXP,"+filePath+":"+predicted[exampleIndices[i].second].file.name+","+QString::number(exampleIndices[i].first));
    }

    for (int i=0; i<pointErrorMagnitudes.size(); i++) {
        QList<float> &pointError = pointErrorMagnitudes[i];
        std::sort(pointError.begin(), pointError.end());
        averagePointErrors.append(Common::Mean(pointError));
        const int keep = qMin(Max_Points, pointError.size());
        for (int j=0; j<keep; j++)
            lines.append(QString("Box,%1,%2").arg(QString::number(i), QString::number(pointError[j*(pointError.size()-1)/(keep-1)])));
    }

    const float averagePointError = Common::Mean(averagePointErrors);

    lines.append(QString("AvgError,0,%1").arg(averagePointError));
    lines.append(QString("NormLength,0,%1").arg(Common::Mean(normalizedLengths)));

    QtUtils::writeFile(csv, lines);

    qDebug("Mean Average Error: %.4f", averagePointError);

    return averagePointError;
}

// Helper class for computing correlation coefficient between two sets of values.
// See Pearson calculation as utilized by python numpy.
class CorrelationCounter
{
private:
    // Variables to compute the correlation
    int num_samples = 0;
    double sumX = 0.f, sumY = 0.f, sumXX = 0.f, sumYY = 0.f, sumXY = 0.f;

    // Variables to compute the 2D histogram correlation
    float gtMinValue = FLT_MAX, gtMaxValue = FLT_MIN, predMinValue = FLT_MAX, predMaxValue = FLT_MIN;

    // Variables to compute the RMS and MAE errors
    float rms = 0.f, mae = 0.f;

    // Hold a list of the scores
    float *gts, *preds;
public:
    CorrelationCounter(int num_samples) {
        this->num_samples = num_samples;
        gts = (float*) malloc(num_samples*sizeof(float));
        preds = (float*) malloc(num_samples*sizeof(float));
    }

    ~CorrelationCounter() {
        printf("\nDestructor\n");
        free(gts);
        free(preds);
        printf("\nEnd Destructor\n");
    }

    void add_sample(int index, double pred, double gt)
    {
        gts[index] = gt;
        preds[index] = pred;

        float difference = pred - gt;
        rms += pow(difference, 2.f) / num_samples;
        mae += fabsf(difference) / num_samples;

        sumX += pred;
        sumY += gt;
        sumXX += pred * pred;
        sumYY += gt * gt;
        sumXY += pred * gt;

        if (gt < gtMinValue) gtMinValue = gt;
        if (gt > gtMaxValue) gtMaxValue = gt;
        if (pred < predMinValue) predMinValue = pred;
        if (pred > predMaxValue) predMaxValue = pred;
    }

    QString getGTValues(QString separator) {
        QString values = "";
        for (int i = 0; i < num_samples - 1; i++)
            values.append(QString("%1%2").arg(QString::number(gts[i])).arg(separator));
        values.append(QString("%1").arg(gts[num_samples-1]));
        return values;
    }

    QString getPredValues(QString separator) {
        QString values = "";
        for (int i = 0; i < num_samples - 1; i++)
            values.append(QString("%1%2").arg(QString::number(preds[i])).arg(separator));
        values.append(QString("%1").arg(gts[num_samples-1]));
        return values;
    }

    int getNumSamples() {
        return num_samples;
    }

    float getRMS() {
        return sqrt(rms);
    }

    float getMAE() {
        return mae;
    }

    double correlation_coefficient()
    {
        double numer = (num_samples * sumXY) - (sumX * sumY);
        double denomX = (num_samples * sumXX) - (sumX * sumX);
        double denomY = (num_samples * sumYY) - (sumY * sumY);
        return numer / sqrt(denomX * denomY);
    }

    void print_hist()
    {
        int histSize = 20;
        unsigned int* hist = new unsigned int[histSize*histSize];
        for (int i = 0; i < histSize * histSize; i++)
            hist[i] = 0;

        printf("\nHistogram Table (PRED = Y-axis in [%.3f,%.3f], GT = X-axis in [%.3f,%.3f])\n| Y // X |", predMinValue, predMaxValue, gtMinValue, gtMaxValue);
        for (int i = 0; i < num_samples; i++) {
            int predBin = MIN(histSize-1, MAX(0, ((int)(histSize*(preds[i]-predMinValue)/(predMaxValue-predMinValue)))));
            int gtBin = MIN(histSize-1, MAX(0, ((int)(histSize*(gts[i]-gtMinValue)/(gtMaxValue-gtMinValue)))));
            hist[histSize * predBin + gtBin] += 1;
        }

        for (int i = 0; i < histSize; i++)
            printf("  %.2f  |", (i / (float) histSize));
        printf("  Total |\n| :----: |");
        for (int i = 0; i < histSize; i++)
            printf(" :----: |");
        printf(" :----: |");
        for (int i = 0; i < histSize; i++) {
            printf("\n|  %.2f  |", (i / (float) histSize));
            float total = 0.f;
            for (int j = 0; j < histSize; j++) {
                float value = hist[i*histSize+j] * (100.f / num_samples);
                total += value;
                if (value == 0)
                    printf("        |");
                else if (value >= 10.f)
                    printf(" %.2f  |", value);
                else
                    printf("  %.2f  |", value);
            }
            if (total == 0)
                printf("        |");
            else if (total >= 10.f)
                printf(" %.2f  |", total);
            else
                printf("  %.2f  |", total);
        }
        printf("\n|  Total |");
        for (int j = 0; j < histSize; j++) {
            float total = 0;
            for (int i = 0; i < histSize; i++) {
                total += hist[i*histSize+j] * (100.f / num_samples);
            }
            if (total == 0)
                printf("        |");
            else if (total >= 10.f)
                printf(" %.2f  |", total);
            else
                printf("  %.2f  |", total);
        }
        printf(" 100.0  |\n\n");

        delete [] hist;
    }
};

void EvalRegression(const TemplateList predicted, const TemplateList truth, QString predictedProperty, QString truthProperty, bool generatePlot) {

    CorrelationCounter *cc = new CorrelationCounter(predicted.size());
    for (int i=0; i<predicted.size(); i++) {
        if (predicted[i].file.name != truth[i].file.name)
            qFatal("Input order mismatch.");

        if (predicted[i].file.contains(predictedProperty) && truth[i].file.contains(truthProperty)) {
            float pred = predicted[i].file.get<float>(predictedProperty);
            float gt = truth[i].file.get<float>(truthProperty);
            cc->add_sample(i, pred, gt);
        }
    }

    if (generatePlot) {
        QStringList rSource;
        rSource << "# Load libraries" << "library(ggplot2)" << "" << "# Set Data"
                << "Actual <- c(" + cc->getGTValues(",") + ")"
                << "Predicted <- c(" + cc->getPredValues(",") + ")"
                << "data <- data.frame(Actual, Predicted)"
                << "" << "# Construct Plot" << "pdf(\"EvalRegression.pdf\")"
                << "print(qplot(Actual, Predicted, data=data, geom=\"jitter\", alpha=I(2/3)) + geom_abline(intercept=0, slope=1, color=\"forestgreen\", size=I(1)) + geom_smooth(size=I(1), color=\"mediumblue\") + theme_bw())"
                << "print(qplot(Actual, Predicted-Actual, data=data, geom=\"jitter\", alpha=I(2/3)) + geom_abline(intercept=0, slope=0, color=\"forestgreen\", size=I(1)) + geom_smooth(size=I(1), color=\"mediumblue\") + theme_bw())"
                << "dev.off()";


        QString rFile = "EvalRegression.R";
        QtUtils::writeFile(rFile, rSource);
        bool success = QtUtils::runRScript(rFile);
        if (success) QtUtils::showFile("EvalRegression.pdf");
    }

    qDebug("Total Samples = %i", cc->getNumSamples());
    qDebug("RMS Error = %f", cc->getRMS());
    qDebug("MAE = %f", cc->getMAE());
    qDebug("Correlation (Pearson) = %f", cc->correlation_coefficient());
    cc->print_hist();
}

void EvalRegression(const QString &predictedGallery, const QString &truthGallery, QString predictedProperty, QString truthProperty, bool generatePlot)
{
    qDebug("Evaluating regression of (%s,%s) against ground truth (%s,%s)", qPrintable(predictedGallery), qPrintable(predictedProperty), qPrintable(truthGallery), qPrintable(truthProperty));

    if (predictedProperty.isEmpty())
        predictedProperty = "Regressor";
    // If predictedProperty is specified, but truthProperty isn't, copy the value over
    // rather than using the default for truthProperty
    else if (truthProperty.isEmpty())
        truthProperty = predictedProperty;

    if (truthProperty.isEmpty())
        predictedProperty = "Regressand";

    const TemplateList predicted(TemplateList::fromGallery(predictedGallery));
    if (predictedGallery == truthGallery) {
        EvalRegression(predicted, predicted, predictedProperty, truthProperty, generatePlot);
    } else {
        const TemplateList truth = TemplateList::fromGallery(truthGallery);
        if (predicted.size() != truth.size()) qFatal("Input size mismatch.");
        EvalRegression(predicted, truth, predictedProperty, truthProperty, generatePlot);
    }
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

    qDebug("FNIR @ FPIR = 0.1:   %.3f", 1-getOperatingPoint(operatingPoints, "FAR", 0.1).TAR);
    qDebug("FNIR @ FPIR = 0.01:  %.3f", 1-getOperatingPoint(operatingPoints, "FAR", 0.01).TAR);
}

void EvalEER(const QString &predictedXML, QString gt_property, QString distribution_property, const QString &csv) {
    if (gt_property.isEmpty())
        gt_property = "Label";
    if (distribution_property.isEmpty())
        distribution_property = "Fusion";
    int classOneTemplateCount = 0;
    int classZeroTemplateCount = 0;
    const TemplateList templateList(TemplateList::fromGallery(predictedXML));

    QList<QPair<float, int>> scores;
    QList<float> classZeroScores, classOneScores;
    float minClassOneScore = std::numeric_limits<float>::max();
    float maxClassOneScore = 0;
    float minClassZeroScore = std::numeric_limits<float>::max();
    float maxClassZeroScore = 0;
    for (int i=0; i<templateList.size(); i++) {
        if (!templateList[i].file.contains(distribution_property) || !templateList[i].file.contains(gt_property))
            continue;

        const int gtLabel = templateList[i].file.get<int>(gt_property);
        const float templateScore = templateList[i].file.get<float>(distribution_property);
        scores.append(qMakePair(templateScore, gtLabel));

        if (gtLabel == 1) {
            classOneTemplateCount++;
            classOneScores.append(templateScore);
            if (minClassOneScore > templateScore)
                minClassOneScore = templateScore;
            if (maxClassOneScore < templateScore)
                maxClassOneScore = templateScore;
        } else {
            classZeroTemplateCount++;
            classZeroScores.append(templateScore);
            if (minClassZeroScore > templateScore)
                minClassZeroScore = templateScore;
            if (maxClassZeroScore < templateScore)
                maxClassZeroScore = templateScore;
        }
    }

    std::sort(scores.begin(), scores.end());

    QList<OperatingPoint> operatingPoints;

    int falsePositives = 0;
    int truePositives = 0;
    size_t index = 0;
    float minDiff = 100, EER = 100, EERThres = 0;
    float threshold = 0;

    while (index < scores.size()) {
        threshold = threshold + 0.001;
        // Compute genuine and imposter statistics at a threshold
        while ((index < scores.size()) && (scores[index].first < threshold)) {
            if (scores[index].second) {
                truePositives++;
            } else {
                falsePositives++;
            }
            index++;
        }

        const float FAR = float(falsePositives) / classZeroTemplateCount;
        const float TAR = float(truePositives) / classOneTemplateCount;
        const float FRR = 1 - TAR;
        operatingPoints.append(OperatingPoint(threshold, FAR, TAR));

        const float diff = std::abs(FAR-FRR);
        if (diff < minDiff) {
            minDiff = diff;
            EER = (FAR+FRR)/2.0;
            EERThres = threshold;
        }
    }

    operatingPoints.append(OperatingPoint(MAX(maxClassOneScore,maxClassZeroScore),1,1));
    operatingPoints.prepend(OperatingPoint(MIN(minClassOneScore,minClassZeroScore),0,0));

    printf("\n==========================================================\n");
    printf("Class 0 Templates: %d\tClass 1 Templates: %d\tTotal Templates: %d\n",
           classZeroTemplateCount, classOneTemplateCount, classZeroTemplateCount + classOneTemplateCount);
    printf("----------------------------------------------------------\n");
    foreach (float FAR, QList<float>() << 0.2 << 0.1 << 0.05 << 0.01 << 0.001 << 0.0001) {
        const OperatingPoint op = getOperatingPoint(operatingPoints, "FAR", FAR);
        if (op.score < minClassOneScore)
            printf("TAR =   0   @ FAR = %.4f | Threshold= %.3f\n", FAR, MIN(op.score, minClassOneScore));
        else if (op.score >= maxClassOneScore)
            printf("TAR =   1   @ FAR = %.4f | Threshold= %.3f\n", FAR, op.score);
        else
            printf("TAR = %.3f @ FAR = %.4f | Threshold= %.3f\n", op.TAR, FAR, op.score);

    }
    printf("----------------------------------------------------------\n");
    foreach (float TAR, QList<float>() << 0.8 << 0.85 << 0.9 << 0.95 << 0.98 << 0.99 << 0.999) {
        const OperatingPoint op = getOperatingPoint(operatingPoints, "TAR", TAR);
        if (op.score < minClassZeroScore)
            printf("FAR =   0   @ TAR = %.4f | Threshold= %.3f\n", TAR, MIN(op.score, minClassZeroScore));
        else if (op.score >= maxClassZeroScore)
            printf("FAR =   0   @ TAR = %.4f | Threshold= %.3f\n", TAR, op.score);
        else
            printf("FAR = %.3f @ TAR = %.4f | Threshold= %.3f\n", op.FAR, TAR, op.score);

    }
    printf("----------------------------------------------------------\n");
    printf("EER: %.3f @ Threshold %.3f\n", EER*100, EERThres);
    printf("==========================================================\n\n");

    QString thresh, form, frr, far;
    int opindex = 0;
    OperatingPoint op = operatingPoints.first();
    foreach (float score, QList<float>() << 0.05 << 0.1 << 0.15 << 0.2 << 0.25 << 0.3 << 0.35 << 0.4 << 0.45 << 0.5
                                               << 0.55 << 0.6 << 0.65 << 0.7 << 0.75 << 0.8 << 0.85 << 0.9 << 0.95) {
        while (opindex < operatingPoints.size() && operatingPoints[opindex].score < score) {
            op.TAR = operatingPoints[opindex].TAR;
            op.FAR = operatingPoints[opindex].FAR;
            opindex++;
        }
        thresh.append(QString("| %1 ").arg(QString::number(score, 'f', 3)));
        form.append(QString("| :---: "));
        frr.append(score > maxClassOneScore ? QString("|       ") : QString("| %1 ").arg(QString::number(std::abs(1.0 - op.TAR), 'f', 3)));
        far.append(score < minClassZeroScore ? QString("|       ") : QString("| %1 ").arg(QString::number(std::abs(op.FAR), 'f', 3)));
    }

    printf("\nClass 0 Templates: %d, Class 1 Templates: %d\nClass 0 Range: [%.3f,%.3f], Class 1 Range: [%.3f,%.3f]\n\n",
        classZeroTemplateCount, classOneTemplateCount, minClassZeroScore, maxClassZeroScore, minClassOneScore, maxClassOneScore);
    printf("|   Threshold    %s|\n", qPrintable(thresh));
    printf("|     :---:      %s|\n", qPrintable(form));
    printf("|  Genuine (FRR) %s|\n", qPrintable(frr));
    printf("|   Spoof  (FAR) %s|\n", qPrintable(far));
    printf("\n\n");

    // Optionally write ROC curve
    if (!csv.isEmpty()) {
        QStringList lines;
        lines.append("Plot,X,Y");
        lines.append("Metadata,"+QString::number(classZeroTemplateCount+classOneTemplateCount)+",Total Templates");
        lines.append("Metadata,"+QString::number(classZeroTemplateCount)+",Class 0 Template Count");
        lines.append("Metadata,"+QString::number(classOneTemplateCount)+",Class 1 Template Count");

        // Write Detection Error Tradeoff (DET), PRE, REC
        float prevFAR = operatingPoints.first().FAR;
        float prevTAR = -1;
        for (int i=0; i<operatingPoints.size(); i++) {
            OperatingPoint operatingPoint = operatingPoints[i];
            if (operatingPoint.FAR == prevFAR || operatingPoint.TAR == prevTAR)
                continue; // Skip until we have the next point
            prevFAR = operatingPoint.FAR;
            prevTAR = operatingPoint.TAR;
            lines.append(QString("DET,%1,%2").arg(QString::number(operatingPoint.FAR),
                                                  QString::number(1-operatingPoint.TAR)));
            lines.append(QString("FAR,%1,%2").arg(QString::number(operatingPoint.score),
                                                  QString::number(operatingPoint.FAR)));
            lines.append(QString("FRR,%1,%2").arg(QString::number(operatingPoint.score),
                                                  QString::number(1-operatingPoint.TAR)));
        }

        // Write TAR@FAR Table (TF)
        foreach (float FAR, QList<float>() << 0.2 << 0.1 << 0.05 << 0.01 << 0.001 << 0.0001)
            lines.append(qPrintable(QString("TF,%1,%2").arg(
                                    QString::number(FAR, 'f'),
                                    QString::number(getOperatingPoint(operatingPoints, "FAR", FAR).TAR, 'f', 3))));

        // Write FAR@TAR Table (FT)
        foreach (float TAR, QList<float>() << 0.8 << 0.85 << 0.9 << 0.95 << 0.98)
            lines.append(qPrintable(QString("FT,%1,%2").arg(
                                        QString::number(TAR, 'f', 2),
                                        QString::number(getOperatingPoint(operatingPoints, "TAR", TAR).FAR, 'f', 3))));

        // Write FAR@Score Table (SF) and TAR@Score table (ST)
        foreach(const float score, QList<float>() << 0.05 << 0.1 << 0.15 << 0.2 << 0.25 << 0.3 << 0.35 << 0.4 << 0.45 << 0.5
                                                  << 0.55 << 0.6 << 0.65 << 0.7 << 0.75 << 0.8 << 0.85 << 0.9 << 0.95) {
            const OperatingPoint op = getOperatingPoint(operatingPoints, "Score", score);
            lines.append(qPrintable(QString("SF,%1,%2").arg(
                                        QString::number(score, 'f', 2),
                                        QString::number(op.FAR))));
            lines.append(qPrintable(QString("ST,%1,%2").arg(
                                        QString::number(score, 'f', 2),
                                        QString::number(op.TAR))));
        }

        // Write FAR/TAR Bar Chart (BC)
        lines.append(qPrintable(QString("BC,0.0001,%1").arg(QString::number(getOperatingPoint(operatingPoints, "FAR", 0.0001).TAR, 'f', 3))));
        lines.append(qPrintable(QString("BC,0.001,%1").arg(QString::number(getOperatingPoint(operatingPoints, "FAR", 0.001).TAR, 'f', 3))));

        // Write SD & KDE
        int points = qMin(qMin(Max_Points, classZeroScores.size()), classOneScores.size());
        if (points > 1) {
            for (int i=0; i<points; i++) {
                float classZeroScore = classZeroScores[double(i) / double(points-1) * double(classZeroScores.size()-1)];
                float classOneScore = classOneScores[double(i) / double(points-1) * double(classOneScores.size()-1)];
                if (classZeroScore == -std::numeric_limits<float>::max()) classZeroScore = minClassZeroScore;
                if (classOneScore == -std::numeric_limits<float>::max()) classOneScore = minClassOneScore;
                lines.append(QString("SD,%1,Genuine").arg(QString::number(classOneScore)));
                lines.append(QString("SD,%1,Impostor").arg(QString::number(classZeroScore)));
            }
        }

        QtUtils::writeFile(csv, lines);
    }
}

} // namespace br
