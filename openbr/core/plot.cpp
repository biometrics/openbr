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

#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFuture>
#include <QList>
#include <QPair>
#include <QPointF>
#include <QRegExp>
#include <QSet>
#include <QStringList>
#include <QVector>
#include <QtAlgorithms>
#include <opencv2/core/core.hpp>
#include <assert.h>

#include "plot.h"
#include "version.h"
#include "openbr/core/bee.h"
#include "openbr/core/common.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/qtutils.h"

#undef FAR // Windows preprecessor definition

using namespace cv;

namespace br
{

void Confusion(const QString &file, float score, int &true_positives, int &false_positives, int &true_negatives, int &false_negatives)
{
    qDebug("Computing confusion matrix of %s at %f", qPrintable(file), score);

    QStringList lines = QtUtils::readLines(file);
    true_positives = false_positives = true_negatives = false_negatives = 0;
    foreach (const QString &line, lines) {
        if (!line.startsWith("SD")) continue;
        QStringList words = line.split(",");
        bool ok;
        float similarity = words[1].toFloat(&ok); assert(ok);
        if (words[2] == "Genuine") {
            if (similarity >= score) true_positives++;
            else                     false_negatives++;
        } else {
            if (similarity >= score) false_positives++;
            else                     true_negatives++;
        }
    }
}

static QStringList getPivots(const QString &file, bool headers)
{
    QString str;
    if (headers) str = QFileInfo(file).dir().dirName();
    else         str = QFileInfo(file).completeBaseName();
    return str.split("_");
}

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

float Evaluate(const QString &simmat, const QString &mask, const QString &csv)
{
    qDebug("Evaluating %s with %s", qPrintable(simmat), qPrintable(mask));

    // Read files
    const Mat scores = BEE::readSimmat(simmat);
    File maskFile(mask);
    maskFile.set("rows", scores.rows);
    maskFile.set("columns", scores.cols);
    const Mat masks = BEE::readMask(maskFile);
    if (scores.size() != masks.size()) qFatal("Simmat (%i,%i) / Mask (%i,%i) size mismatch.", scores.rows, scores.cols, masks.rows, masks.cols);

    return Evaluate(scores, masks, csv);
}

float Evaluate(const Mat &simmat, const Mat &mask, const QString &csv)
{
    const int Max_Points = 500;
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

    double genuineSum = 0, impostorSum = 0;
    QList<OperatingPoint> operatingPoints;
    QList<float> genuines, impostors;
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
                genuineSum += comparison.score;
                genuines.append(comparison.score);
                if (firstGenuineReturns[comparison.query] < 1)
                    firstGenuineReturns[comparison.query] = abs(firstGenuineReturns[comparison.query]) + 1;
                if ((comparison.score != -std::numeric_limits<float>::max()) &&
                    (comparison.score < minGenuineScore))
                    minGenuineScore = comparison.score;
            } else {
                falsePositives++;
                impostorSum += comparison.score;
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
            // Restrict the extreme ends of the curve
            if ((falsePositives >= 10) && (falsePositives < impostorCount/2))
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

    const double hGenuine = Common::KernelDensityBandwidth(sampledGenuineScores);
    foreach (double f, sampledGenuineScores)
        lines.append(QString("KDEGenuine,%1,%2").arg(QString::number(f), QString::number(Common::KernelDensityEstimation(sampledGenuineScores, f, hGenuine))));

    const double hImpostor = Common::KernelDensityBandwidth(sampledImpostorScores);
    foreach (double f, sampledImpostorScores)
        lines.append(QString("KDEImpostor,%1,%2").arg(QString::number(f), QString::number(Common::KernelDensityEstimation(sampledImpostorScores, f, hImpostor))));

    // Write Cumulative Match Characteristic (CMC) curve
    const int Max_Retrieval = 100;
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

    if (!csv.isEmpty()) QtUtils::writeFile(csv, lines);
    qDebug("TAR @ FAR = 0.01: %.3f\nRetrieval Rate @ Rank = %d: %.3f", result, Report_Retrieval, reportRetrievalRate);
    return result;
}

static QString getScale(const QString &mode, const QString &title, int vals)
{
    if      (vals > 12) return " + scale_"+mode+"_discrete(\""+title+"\")";
    else if (vals > 11) return " + scale_"+mode+"_brewer(\""+title+"\", palette=\"Set3\")";
    else if (vals > 9)  return " + scale_"+mode+"_brewer(\""+title+"\", palette=\"Paired\")";
    else                return " + scale_"+mode+"_brewer(\""+title+"\", palette=\"Set1\")";
}

// Custom sorting method to ensure datasets are ordered nicely
static bool sortFiles(const QString &fileA, const QString &fileB)
{
    QFileInfo fileInfoA(fileA);
    QFileInfo fileInfoB(fileB);

    if (fileInfoA.fileName().contains("Good")) return true;
    if (fileInfoB.fileName().contains("Good")) return false;
    if (fileInfoA.fileName().contains("Bad")) return true;
    if (fileInfoB.fileName().contains("Bad")) return false;
    if (fileInfoA.fileName().contains("Ugly")) return true;
    if (fileInfoB.fileName().contains("Ugly")) return false;
    if (fileInfoA.fileName().contains("MEDS")) return true;
    if (fileInfoB.fileName().contains("MEDS")) return false;
    if (fileInfoA.fileName().contains("PCSO")) return true;
    if (fileInfoB.fileName().contains("PCSO")) return false;

    return fileA < fileB;
}

struct RPlot
{
    QString basename, suffix;
    QFile file;
    QStringList pivotHeaders;
    QVector< QSet<QString> > pivotItems;
    bool flip;

    struct Pivot
    {
        int index, size;
        QString header;
        bool smooth;
        Pivot() : index(-1), size(0), smooth(false) {}
        Pivot(int _index, int _size, const QString &_header)
            : index(_index), size(_size), header(_header), smooth(false) {}
    };

    Pivot major, minor;

    RPlot(QStringList files, const br::File &destination, bool isEvalFormat = true)
    {
        if (files.isEmpty()) qFatal("Empty file list.");
        qSort(files.begin(), files.end(), sortFiles);

        // Parse destination
        QFileInfo fileInfo(destination);
        basename = fileInfo.path() + "/" + fileInfo.completeBaseName();
        suffix = fileInfo.suffix();
        if (suffix.isEmpty()) suffix = "pdf";

        file.setFileName(basename+".R");
        bool success = file.open(QFile::WriteOnly);
        if (!success) qFatal("Failed to open %s for writing.", qPrintable(file.fileName()));

        file.write("# Load libraries\n"
                   "library(ggplot2)\n"
                   "library(gplots)\n"
                   "library(reshape)\n"
                   "library(scales)\n"
                   "\n"
                   "# Read CSVs\n"
                   "data <- NULL\n");

        // Read files and retrieve pivots
        pivotHeaders = getPivots(files.first(), true);
        pivotItems = QVector< QSet<QString> >(pivotHeaders.size());
        foreach (const QString &fileName, files) {
            QStringList pivots = getPivots(fileName, false);
            
            // If the number of pivots don't match, abandon the directory/filename labeling scheme
            if (pivots.size() != pivotHeaders.size()) {
                pivots.clear();
                pivots.push_back(QFileInfo(fileName).completeBaseName());
                pivotHeaders.clear();
                pivotHeaders.push_back("File");
            } 
            file.write(qPrintable(QString("tmp <- read.csv(\"%1\")\n").arg(fileName).replace("\\", "\\\\")));
            for (int i=0; i<pivots.size(); i++) {
                pivotItems[i].insert(pivots[i]);
                file.write(qPrintable(QString("tmp$%1 <- \"%2\"\n").arg(pivotHeaders[i], pivots[i])));
            }
            file.write("data <- rbind(data, tmp)\n");
        }

        // Format data
        if (isEvalFormat)
            file.write("\n"
                       "# Split data into individual plots\n"
                       "plot_index = which(names(data)==\"Plot\")\n"
                       "Metadata <- data[grep(\"Metadata\",data$Plot),-c(1)]\n"
                       "DET <- data[grep(\"DET\",data$Plot),-c(1)]\n"
                       "FAR <- data[grep(\"FAR\",data$Plot),-c(1)]\n"
                       "FRR <- data[grep(\"FRR\",data$Plot),-c(1)]\n"
                       "SD <- data[grep(\"SD\",data$Plot),-c(1)]\n"
                       "BC <- data[grep(\"BC\",data$Plot),-c(1)]\n"
                       "CMC <- data[grep(\"CMC\",data$Plot),-c(1)]\n"
                       "FAR$Error <- \"FAR\"\n"
                       "FRR$Error <- \"FRR\"\n"
                       "ERR <- rbind(FAR, FRR)\n"
                       "rm(data, FAR, FRR)\n"
                       "\n"
                       "# Format data\n"
                       "Metadata$Y<-factor(Metadata$Y, levels=c(\"Genuine\",\"Impostor\",\"Ignored\",\"Gallery\",\"Probe\"))\n"
                       "DET$Y <- as.numeric(as.character(DET$Y))\n"
                       "ERR$Y <- as.numeric(as.character(ERR$Y))\n"
                       "SD$Y <- as.factor(unique(as.character(SD$Y)))\n"
                       "BC$Y <- as.numeric(as.character(BC$Y))\n"
                       "CMC$Y <- as.numeric(as.character(CMC$Y))\n"
                       "\n"
                       "# Code to format FAR values\n"
                       "far_names <- list('0.001'=\"FAR = 0.1%\", '0.01'=\"FAR = 1%\")\n"
                       "far_labeller <- function(variable,value) { return(far_names[as.character(value)]) }\n");

        // Open output device
        file.write(qPrintable(QString("\n"
                                      "# Open output device\n"
                                      "%1(\"%2.%1\")\n").arg(suffix, basename)));

        // Write metadata table
        if ((suffix == "pdf") && isEvalFormat) {
            file.write("\n"
                       "# Write metadata table\n");
            QString MT = "MT <- as.data.frame(recast(Metadata, %1 ~ Y, id.var=c(\"%2\",\"Y\"), measure.var=c(\"X\")))\n";
            file.write(qPrintable(MT.arg(pivotHeaders.join(" + "), pivotHeaders.join("\",\""))));
            QString textplot = "par(mfrow=c(4,1))\n"
                               "plot.new()\n"
                               "print(title(paste(\"%1 - %2\",date(),sep=\"\\n\")))\n"
                               "print(textplot(MT[,!(names(MT) %in% c(\"Genuine\", \"Impostor\", \"Ignored\"))], show.rownames=FALSE))\n"
                               "print(title(\"Images\"))\n"
                               "print(textplot(MT[,!(names(MT) %in% c(\"Gallery\", \"Probe\"))], show.rownames=FALSE))\n"
                               "print(title(\"Matches\"))\n"
                               "plot.new()\n"
                               "print(title(\"Gallery * Probe = Genuine + Impostor + Ignored\"))\n";
            file.write(qPrintable(textplot.arg(PRODUCT_NAME, PRODUCT_VERSION)));
        }

        // Write figures
        file.write("\n"
                   "# Write figures\n");

        for (int i=0; i<pivotItems.size(); i++) {
            const int size = pivotItems[i].size();
            if (size > major.size) {
                minor = major;
                major = Pivot(i, size, pivotHeaders[i]);
            } else if (size > minor.size) {
                minor = Pivot(i, size, pivotHeaders[i]);
            }
        }

        const QString &smooth = destination.get<QString>("smooth", "");
        major.smooth = !smooth.isEmpty() && (major.header == smooth) && (major.size > 1);
        minor.smooth = !smooth.isEmpty() && (minor.header == smooth) && (minor.size > 1);
        if (major.smooth) major.size = 1;
        if (minor.smooth) minor.size = 1;
        if (major.size < minor.size)
            std::swap(major, minor);

        flip = minor.header == "Algorithm";
    }

    bool finalize(bool show = false)
    {
        file.write("dev.off()\n");
        if (suffix != "pdf") file.write(qPrintable(QString("unlink(\"%1.%2\")").arg(basename, suffix)));
        file.close();

        bool success = QtUtils::runRScript(file.fileName());
        if (success && show) QtUtils::showFile(basename+"."+suffix);
        return success;
    }

    QString subfile(const QString &name) const
    {
        return basename+"_"+name+"."+suffix;
    }
};

bool Plot(const QStringList &files, const br::File &destination, bool show)
{
    qDebug("Plotting %d file(s) to %s", files.size(), qPrintable(destination));

    const bool minimalist = destination.getBool("minimalist");

    RPlot p(files, destination);

    p.file.write(qPrintable(QString("qplot(X, 1-Y, data=DET%1").arg((p.major.smooth || p.minor.smooth) ? ", geom=\"smooth\", method=loess, level=0.99" : ", geom=\"line\"") +
                            (p.major.size > 1 ? QString(", colour=factor(%1)").arg(p.major.header) : QString()) +
                            (p.minor.size > 1 ? QString(", linetype=factor(%1)").arg(p.minor.header) : QString()) +
                            QString(", xlab=\"False Accept Rate\", ylab=\"True Accept Rate\") + theme_minimal()") +
                            (p.major.size > 1 ? getScale("colour", p.major.header, p.major.size) : QString()) +
                            (p.minor.size > 1 ? QString(" + scale_linetype_discrete(\"%1\")").arg(p.minor.header) : QString()) +
                            QString(" + scale_x_log10(labels=percent) + scale_y_continuous(labels=percent) + annotation_logticks(sides=\"b\")") +
                            QString("\nggsave(\"%1\")\n\n").arg(p.subfile("ROC"))));

    p.file.write(qPrintable(QString("qplot(X, Y, data=DET%1").arg((p.major.smooth || p.minor.smooth) ? ", geom=\"smooth\", method=loess, level=0.99" : ", geom=\"line\"") +
                            (p.major.size > 1 ? QString(", colour=factor(%1)").arg(p.major.header) : QString()) +
                            (p.minor.size > 1 ? QString(", linetype=factor(%1)").arg(p.minor.header) : QString()) +
                            QString(", xlab=\"False Accept Rate\", ylab=\"False Reject Rate\") + geom_abline(alpha=0.5, colour=\"grey\", linetype=\"dashed\") + theme_minimal()") +
                            (p.major.size > 1 ? getScale("colour", p.major.header, p.major.size) : QString()) +
                            (p.minor.size > 1 ? QString(" + scale_linetype_discrete(\"%1\")").arg(p.minor.header) : QString()) +
                            QString(" + scale_x_log10(labels=percent) + scale_y_log10(labels=percent) + annotation_logticks()") +
                            QString("\nggsave(\"%1\")\n\n").arg(p.subfile("DET"))));

    p.file.write(qPrintable(QString("qplot(X, data=SD, geom=\"histogram\", fill=Y, position=\"identity\", alpha=I(1/2)") +
                            QString(", xlab=\"Score%1\"").arg((p.flip ? p.major.size : p.minor.size) > 1 ? " / " + (p.flip ? p.major.header : p.minor.header) : QString()) +
                            QString(", ylab=\"Frequency%1\"").arg((p.flip ? p.minor.size : p.major.size) > 1 ? " / " + (p.flip ? p.minor.header : p.major.header) : QString()) +
                            QString(") + scale_fill_manual(\"Ground Truth\", values=c(\"blue\", \"red\")) + theme_minimal() + scale_x_continuous(minor_breaks=NULL) + scale_y_continuous(minor_breaks=NULL) + theme(axis.text.y=element_blank(), axis.ticks=element_blank(), axis.text.x=element_text(angle=-90, hjust=0))") +
                            (p.major.size > 1 ? (p.minor.size > 1 ? QString(" + facet_grid(%2 ~ %1, scales=\"free\")").arg((p.flip ? p.major.header : p.minor.header), (p.flip ? p.minor.header : p.major.header)) : QString(" + facet_wrap(~ %1, scales = \"free\")").arg(p.major.header)) : QString()) +
                            QString(" + theme(aspect.ratio=1)") +
                            QString("\nggsave(\"%1\")\n\n").arg(p.subfile("SD"))));

    p.file.write(qPrintable(QString("ggplot(CMC, aes(x=X, y=Y%1%2)) + xlab(\"Rank\") + ylab(\"Retrieval Rate\")").arg(p.major.size > 1 ? QString(" ,colour=factor(%1)").arg(p.major.header) : QString(), p.minor.size > 1 ? QString(", linetype=factor(%1)").arg(p.minor.header) : QString()) +
                            ((p.major.smooth || p.minor.smooth) ? (minimalist ? " + stat_summary(geom=\"line\", fun.y=mean)" : " + stat_summary(geom=\"line\", fun.y=min, aes(linetype=\"Min/Max\")) + stat_summary(geom=\"line\", fun.y=max, aes(linetype=\"Min/Max\")) + stat_summary(geom=\"line\", fun.y=mean, aes(linetype=\"Mean\")) + scale_linetype_manual(\"Legend\", values=c(\"Mean\"=1, \"Min/Max\"=2))") : " + geom_line()") +
                            (minimalist ? "" : "+ scale_x_log10(labels=c(1,5,10,50,100), breaks=c(1,5,10,50,100)) + annotation_logticks(sides=\"b\")") +
                            (p.major.size > 1 ? getScale("colour", p.major.header, p.major.size) : QString()) +
                            (p.minor.size > 1 ? QString(" + scale_linetype_discrete(\"%1\")").arg(p.minor.header) : QString()) +
                            QString(" + theme_minimal() + scale_y_continuous(labels=percent)") +
                            QString("\nggsave(\"%1\")\n\n").arg(p.subfile("CMC"))));

    p.file.write(qPrintable(QString("qplot(factor(%1)%2, data=BC, %3").arg(p.major.smooth ? (p.minor.header.isEmpty() ? "Algorithm" : p.minor.header) : p.major.header, (p.major.smooth || p.minor.smooth) ? ", Y" : "", (p.major.smooth || p.minor.smooth) ? "geom=\"boxplot\"" : "geom=\"bar\", position=\"dodge\", weight=Y") +
                            (p.major.size > 1 ? QString(", fill=factor(%1)").arg(p.major.header) : QString()) +
                            QString(", xlab=\"%1False Accept Rate\"").arg(p.major.size > 1 ? p.major.header + " / " : QString()) +
                            QString(", ylab=\"True Accept Rate%1\") + theme_minimal()").arg(p.minor.size > 1 ? " / " + p.minor.header : QString()) +
                            (p.major.size > 1 ? getScale("fill", p.major.header, p.major.size) : QString()) +
                            (p.minor.size > 1 ? QString(" + facet_grid(%2 ~ X)").arg(p.minor.header) : QString(" + facet_grid(. ~ X, labeller=far_labeller)")) +
                            QString(" + scale_y_continuous(labels=percent) + theme(legend.position=\"none\", axis.text.x=element_text(angle=-90, hjust=0))%1").arg((p.major.smooth || p.minor.smooth) ? "" : " + geom_text(data=BC, aes(label=Y, y=0.05))") +
                            QString("\nggsave(\"%1\")\n\n").arg(p.subfile("BC"))));

    p.file.write(qPrintable(QString("qplot(X, Y, data=ERR%1, linetype=Error").arg((p.major.smooth || p.minor.smooth) ? ", geom=\"smooth\", method=loess, level=0.99" : ", geom=\"line\"") +
                            ((p.flip ? p.major.size : p.minor.size) > 1 ? QString(", colour=factor(%1)").arg(p.flip ? p.major.header : p.minor.header) : QString()) +
                            QString(", xlab=\"Score%1\", ylab=\"Error Rate\") + theme_minimal()").arg((p.flip ? p.minor.size : p.major.size) > 1 ? " / " + (p.flip ? p.minor.header : p.major.header) : QString()) +
                            ((p.flip ? p.major.size : p.minor.size) > 1 ? getScale("colour", p.flip ? p.major.header : p.minor.header, p.flip ? p.major.size : p.minor.size) : QString()) +
                            QString(" + scale_y_log10(labels=percent) + annotation_logticks(sides=\"l\")") +
                            ((p.flip ? p.minor.size : p.major.size) > 1 ? QString(" + facet_wrap(~ %1, scales=\"free_x\")").arg(p.flip ? p.minor.header : p.major.header) : QString()) +
                            QString(" + theme(aspect.ratio=1)") +
                            QString("\nggsave(\"%1\")\n\n").arg(p.subfile("ERR"))));

    return p.finalize(show);
}

bool PlotMetadata(const QStringList &files, const QString &columns, bool show)
{
    qDebug("Plotting %d metadata file(s) for columns %s", files.size(), qPrintable(columns));

    RPlot p(files, "PlotMetadata", false);
    foreach (const QString &column, columns.split(";"))
        p.file.write(qPrintable(QString("qplot(%1, %2, data=data, geom=\"violin\", fill=%1) + coord_flip() + theme_minimal()\nggsave(\"%2.pdf\")\n").arg(p.major.header, column)));
    return p.finalize(show);
}

} // namespace br
