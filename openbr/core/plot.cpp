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

#include "plot.h"
#include "version.h"
#include "openbr/core/qtutils.h"

using namespace cv;

namespace br
{

// Flattens file metadata to an R list()
static QString toRList(const File &opts)
{
    QStringList retValues;
    QString format = "%1=%2";
    foreach (const QString &key, opts.localKeys()) {
        const QString value = QtUtils::toString(opts.value(key));
        if (value.startsWith("("))
            retValues.append(format.arg(key, "\"c" + value + "\""));
        else if (value == "true")
            retValues.append(format.arg(key, "TRUE"));
        else if (value == "false")
            retValues.append(format.arg(key, "FALSE"));
        else
            retValues.append(format.arg(key, "\"" + value + "\""));
    }
    return retValues.join(",");
}

static QStringList getPivots(const QString &file, bool headers)
{
    QString str;
    if (headers) str = QFileInfo(file).dir().dirName();
    else         str = QFileInfo(file).completeBaseName();
    return str.split("_");
}

// Custom sorting method to ensure datasets are ordered nicely
static bool sortFiles(const QString &fileA, const QString &fileB)
{
    return fileA < fileB;
}

struct RPlot
{
    QString basename, suffix;
    QFile file;
    QStringList pivotHeaders;
    QVector< QSet<QString> > pivotItems;

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

    RPlot(QStringList files, const File &destination)
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

        // Copy plot_utils.R into output script with source()
        QString plotUtilsPath = Globals->sdkPath + "/openbr/share/openbr/plotting/plot_utils.R"; // Handle case when OpenBR is a submodule
        if (!QFileInfo(plotUtilsPath).exists())
            plotUtilsPath = Globals->sdkPath + "/share/openbr/plotting/plot_utils.R";
        file.write(qPrintable(QString("source(\"%1\")\n\n").arg(plotUtilsPath)));
        file.write("# Read CSVs\n"
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

        // Set variables in R
        file.write(qPrintable(QString("\nconfidence <- %1\n").arg(destination.get<float>("confidence", 95) / 100.0)));
        file.write(qPrintable(QString("ncol <- %1\n").arg(destination.get<int>("ncol", major.size > 1 ? major.size : (minor.header.isEmpty() ? major.size : minor.size)))));
        file.write(qPrintable(QString("seq <- %1\n").arg(destination.get<bool>("seq", false) ? "TRUE": "FALSE")));
        file.write(qPrintable(QString("basename <- \"%1\"\n").arg(basename)));
        file.write(qPrintable(QString("smooth <- %1\n").arg((major.smooth || minor.smooth) && (destination.get<float>("confidence", 95) / 100.0) != 0 ? "TRUE" : "FALSE")));
        file.write(qPrintable(QString("csv <- %1\n").arg(destination.getBool("csv") ? "TRUE" : "FALSE")));
        file.write(qPrintable(QString("majorHeader <- \"%1\"\n").arg(major.header)));
        file.write(qPrintable(QString("majorSize <- %1\n").arg(major.size)));
        file.write(qPrintable(QString("majorSmooth <- %1\n").arg(major.smooth ? "TRUE" : "FALSE")));
        file.write(qPrintable(QString("minorHeader <- \"%1\"\n").arg(minor.header)));
        file.write(qPrintable(QString("minorSize <- %1\n").arg(minor.size)));
        file.write(qPrintable(QString("minorSmooth <- %1\n").arg(minor.smooth ? "TRUE" : "FALSE")));
        file.write(qPrintable(QString("flip <- %1\n").arg(minor.header == "Algorithm" ? "TRUE" : "FALSE")));

        // Open output device
        file.write(qPrintable(QString("\n"
                                      "# Open output device\n"
                                      "%1(\"%2.%1\"%3)\n").arg(suffix, basename, suffix != "pdf" ? ", width=800, height=800" : "")));

        // Write figures
        file.write("\n"
                   "# Write figures\n");
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
};

// Does not work if dataset folder starts with a number
bool Plot(const QStringList &files, const File &destination, bool show)
{
    qDebug("Plotting %d file(s) to %s", files.size(), qPrintable(destination));

    RPlot p(files, destination);
    p.file.write("\nformatData()\n\n");
    p.file.write(qPrintable(QString("algs <- %1\n").arg((p.major.size > 1 && p.minor.size > 1) && !(p.major.smooth || p.minor.smooth) ? QString("paste(TF$%1, TF$%2, sep=\"_\")").arg(p.major.header, p.minor.header)
                                                                                                                                      : QString("TF$%1").arg(p.major.size > 1 ? p.major.header : (p.minor.header.isEmpty() ? p.major.header : p.minor.header)))));
    p.file.write("algs <- algs[!duplicated(algs)]\n");

    if (p.major.smooth || p.minor.smooth) {
        QString groupvar = p.major.size > 1 ? p.major.header : (p.minor.header.isEmpty() ? p.major.header : p.minor.header);
        foreach(const QString &data, QStringList() << "DET" << "IET" << "CMC" << "TF" << "FT" << "CT") {
            p.file.write(qPrintable(QString("%1 <- summarySE(%1, measurevar=\"Y\", groupvars=c(\"%2\", \"X\"), conf.interval=confidence)"
                                            "\n").arg(data, groupvar)));
        }
        p.file.write(qPrintable(QString("%1 <- summarySE(%1, measurevar=\"X\", groupvars=c(\"Error\", \"%2\", \"Y\"), conf.interval=confidence)"
                                        "\n\n").arg("ERR", groupvar)));
    }

    // Use a br::file for simple storage of plot options
    QMap<QString,File> optMap;
    optMap.insert("rocOptions", File(QString("[xTitle=False Accept Rate,yTitle=True Accept Rate,xLog=true,yLog=false,xLimits=(.0000001,.1)]")));
    optMap.insert("detOptions", File(QString("[xTitle=False Accept Rate,yTitle=False Reject Rate,xLog=true,yLog=true,xLimits=(.0000001,.1),yLimits=(.0001,1)]")));
    optMap.insert("ietOptions", File(QString("[xTitle=False Positive Identification Rate (FPIR),yTitle=False Negative Identification Rate (FNIR),xLog=true,yLog=true]")));
    optMap.insert("cmcOptions", File(QString("[xTitle=Rank,yTitle=Retrieval Rate,xLog=true,yLog=false,size=1,xLabels=(1,5,10,50,100),xBreaks=(1,5,10,50,100)]")));
    optMap.insert("farOptions", File(QString("[xTitle=Score,yTitle=False Accept Rate,xLog=false,yLog=true,xLabels=waiver(),yLimits=(.0000001,1)]")));
    optMap.insert("frrOptions", File(QString("[xTitle=Score,yTitle=False Reject Rate,xLog=false,yLog=true,xLabels=waiver(),yLimits=(.0001,1)]")));

    foreach (const QString &key, optMap.keys()) {
        const QStringList options = destination.get<QStringList>(key, QStringList());
        foreach (const QString &option, options) {
            QStringList words = QtUtils::parse(option, '=');
            QtUtils::checkArgsSize(words[0], words, 1, 2);
            optMap[key].set(words[0], words[1]);
        }
    }

    // optional plot metadata and accuracy tables
    if (destination.getBool("metadata", true)) {
        p.file.write("plotTAR(tableData=TF)\n");
        p.file.write("plotTAR(tableData=TF,operatingPoint=1e-6)\n");
    }

    // Write plots
    QString plot = "plot <- plotLine(lineData=%1, options=list(%2), flipY=%3)\nplot\n";
    p.file.write(qPrintable(QString(plot).arg("DET", toRList(optMap["rocOptions"]), "TRUE")));
    p.file.write(qPrintable(QString(plot).arg("DET", toRList(optMap["detOptions"]), "FALSE")));
    p.file.write(qPrintable(QString(plot).arg("IET", toRList(optMap["ietOptions"]), "FALSE")));
    p.file.write(qPrintable(QString(plot).arg("CMC", toRList(optMap["cmcOptions"]), "FALSE")));
    p.file.write("plot <- plotSD(sdData=SD)\nplot\n");
    p.file.write("plot <- plotBC(bcData=BC)\nplot\n");
    p.file.write(qPrintable(QString(plot).arg("FAR", toRList(optMap["farOptions"]), "FALSE")));
    p.file.write(qPrintable(QString(plot).arg("FRR", toRList(optMap["frrOptions"]), "FALSE")));
    p.file.write("plotEERSamples(imData=IM, gmData=GM)\n\n");

    return p.finalize(show);
}

//Check if only one ROC point is in the file
bool fileHasSinglePoint(const QString &evalFile) {
    QFile file(evalFile);
    bool success = file.open(QFile::ReadOnly);
    if (!success) qFatal("Failed to open %s for reading.", qPrintable(evalFile));
    QStringList lines = QString(file.readAll()).split("\n");
    file.close();

    int rocCnt = 0;
    foreach (const QString &line, lines) {
        if (line.contains("DiscreteROC")) {
            rocCnt++;
        }
        if (rocCnt > 1)
            return false;
    }

    return true;
}

//Check all files to see if any single file has only have one ROC point
bool filesHaveSinglePoint(const QStringList &files) {
    foreach (const File &file, files)
        if (fileHasSinglePoint(file))
            return true;
    return false;
}

// Properly

bool PlotDetection(const QStringList &files, const File &destination, bool show)
{
    qDebug("Plotting %d detection file(s) to %s", files.size(), qPrintable(destination));
    RPlot p(files, destination);
    p.file.write("\nformatData(type=\"detection\")\n\n");

    // Use a br::file for simple storage of plot options
    QMap<QString,File> optMap;
    optMap.insert("rocOptions", File(QString("[xTitle=False Accepts Per Image,yTitle=True Accept Rate,xLog=true,yLog=false]")));
    optMap.insert("prOptions", File(QString("[xTitle=Recall,yTitle=Precision,xLog=false,yLog=false]")));

    foreach (const QString &key, optMap.keys()) {
        const QStringList options = destination.get<QStringList>(key, QStringList());
        foreach (const QString &option, options) {
            QStringList words = QtUtils::parse(option, '=');
            QtUtils::checkArgsSize(words[0], words, 1, 2);
            optMap[key].set(words[0], words[1]);
        }
    }

    QString plotType("line");
    if (filesHaveSinglePoint(files))
        plotType = QString("point");

    QString plot = "plot <- plotLine(lineData=%1, options=list(%2), flipY=%3, geometry=%4)\nplot\n";
    foreach (const QString &type, QStringList() << "Discrete" << "Continuous") {
        optMap["rocOptions"].set("title", type);
        if (type == "Continuous")
            optMap["rocOptions"].set("xLimits", QString("(.005,1)"));
        p.file.write(qPrintable(QString(plot).arg(type + "ROC", toRList(optMap["rocOptions"]), "FALSE", "\"" + plotType + "\"")));
    }

    foreach (const QString &type, QStringList() << "Discrete" << "Continuous") {
        optMap["prOptions"].set("title", type);
        p.file.write(qPrintable(QString(plot).arg(type + "PR", toRList(optMap["prOptions"]), "FALSE", "\"" + plotType + "\"")));
    }
    p.file.write("plot <- plotOverlap(overlapData=Overlap)\nplot\n");

    p.file.write(qPrintable(QString("ggplot(AverageOverlap, aes(x=%1, y=%2, label=round(X,3)), main=\"Average Overlap\") + geom_text() + theme_minimal()").arg(p.minor.size > 1 ? p.minor.header : "'X'", p.major.size > 1 ? p.major.header : "'Y'") +
                            QString("%1%2\n\n").arg(p.minor.size > 1 ? "" : " + xlab(NULL)", p.major.size > 1 ? "" : " + ylab(NULL)")));

    p.file.write(qPrintable(QString("ggplot(AverageOverlap, aes(x=%1, y=%2, fill=X)) + geom_tile() + scale_fill_continuous(\"Average Overlap\", guide=FALSE) + theme_minimal()").arg(p.minor.size > 1 ? p.minor.header : "'X'", p.major.size > 1 ? p.major.header : "'Y'") +
                            QString("%1%2\n\n").arg(p.minor.size > 1 ? "" : " + xlab(NULL)", p.major.size > 1 ? "" : " + ylab(NULL)")));

    return p.finalize(show);
}

bool PlotLandmarking(const QStringList &files, const File &destination, bool show)
{
    qDebug("Plotting %d landmarking file(s) to %s", files.size(), qPrintable(destination));
    RPlot p(files, destination);
    p.file.write("\nformatData(type=\"landmarking\")\n\n");
    p.file.write(qPrintable(QString("algs <- unique(Box$%1)\n").arg(p.major.size > 1 ? p.major.header : (p.minor.header.isEmpty() ? p.major.header : p.minor.header))));
    p.file.write("algs <- algs[!duplicated(algs)]\n");
    p.file.write("plotLandmarkSamples(displaySample=displaySample, expData=EXP, extData=EXT)\n");
    p.file.write("plotLandmarkTables(tableData=Box)\n");

    p.file.write(qPrintable(QString("ggplot(Box, aes(Y,%1%2))").arg(p.major.size > 1 ? QString(", colour=%1").arg(p.major.header) : QString(),
                                                                    p.minor.size > 1 ? QString(", linetype=%1").arg(p.minor.header) : QString()) +
                            QString(" + annotation_logticks(sides=\"b\") + stat_ecdf() + scale_x_log10(\"Normalized Error\", breaks=c(0.001,0.01,0.1,1,10)) + scale_y_continuous(\"Cumulative Density\", label=percent) + theme_minimal()\n\n")));

    p.file.write(qPrintable(QString("ggplot(Box, aes(factor(X), Y%1%2))").arg(p.major.size > 1 ? QString(", colour=%1").arg(p.major.header) : QString(), p.minor.size > 1 ? QString(", linetype=%1").arg(p.minor.header) : QString()) +
                            QString("+ annotation_logticks(sides=\"l\") + geom_boxplot(alpha=0.5) + geom_jitter(size=1, alpha=0.5) + scale_x_discrete(\"Landmark\") + scale_y_log10(\"Normalized Error\", breaks=c(0.001,0.01,0.1,1,10)) + theme_minimal()\n\n")));

    return p.finalize(show);
}

bool PlotMetadata(const QStringList &files, const QString &columns, bool show)
{
    qDebug("Plotting %d metadata file(s) for columns %s", files.size(), qPrintable(columns));

    RPlot p(files, "PlotMetadata");
    foreach (const QString &column, columns.split(";"))
        p.file.write(qPrintable(QString("qplot(%1, %2, data=data, geom=\"violin\", fill=%1) + coord_flip() + theme_minimal()\nggsave(\"%2.pdf\")\n").arg(p.major.header, column)));
    return p.finalize(show);
}

bool PlotKNN(const QStringList &files, const File &destination, bool show)
{
    qDebug("Plotting %d k-NN file(s) to %s", files.size(), qPrintable(destination));
    RPlot p(files, destination);
    p.file.write("\nformatData(type=\"knn\")\n\n");

    QMap<QString,File> optMap;
    optMap.insert("rocOptions", File(QString("[xTitle=False Positive Identification Rate (FPIR),yTitle=True Positive Identification Rate (TPIR),xLog=true,yLog=false]")));
    optMap.insert("ietOptions", File(QString("[xTitle=False Positive Identification Rate (FPIR),yTitle=False Negative Identification Rate (FNIR),xLog=true,yLog=true]")));
    optMap.insert("cmcOptions", File(QString("[xTitle=Rank,yTitle=Retrieval Rate,xLog=true,yLog=false,size=1,xLabels=(1,5,10,50,100),xBreaks=(1,5,10,50,100)]")));

    foreach (const QString &key, optMap.keys()) {
        const QStringList options = destination.get<QStringList>(key, QStringList());
        foreach (const QString &option, options) {
            QStringList words = QtUtils::parse(option, '=');
            QtUtils::checkArgsSize(words[0], words, 1, 2);
            optMap[key].set(words[0], words[1]);
        }
    }

    QString plot = "plot <- plotLine(lineData=%1, options=list(%2), flipY=%3)\nplot\n";
    p.file.write(qPrintable(QString(plot).arg("IET", toRList(optMap["rocOptions"]), "TRUE")));
    p.file.write(qPrintable(QString(plot).arg("IET", toRList(optMap["ietOptions"]), "FALSE")));
    p.file.write(qPrintable(QString(plot).arg("CMC", toRList(optMap["cmcOptions"]), "FALSE")));

    return p.finalize(show);
}

// Does not work if dataset folder starts with a number
bool PlotEER(const QStringList &files, const File &destination, bool show)
{
    qDebug("Plotting %d file(s) to %s", files.size(), qPrintable(destination));

    RPlot p(files, destination);
    p.file.write("\nformatData()\n\n");
    p.file.write(qPrintable(QString("algs <- %1\n").arg((p.major.size > 1 && p.minor.size > 1) && !(p.major.smooth || p.minor.smooth) ? QString("paste(TF$%1, TF$%2, sep=\"_\")").arg(p.major.header, p.minor.header)
                                                                                                                                      : QString("TF$%1").arg(p.major.size > 1 ? p.major.header : (p.minor.header.isEmpty() ? p.major.header : p.minor.header)))));
    p.file.write("algs <- algs[!duplicated(algs)]\n");
    if (p.major.smooth || p.minor.smooth) {
        QString groupvar = p.major.size > 1 ? p.major.header : (p.minor.header.isEmpty() ? p.major.header : p.minor.header);
        foreach(const QString &data, QStringList() << "DET" << "TF" << "FT") {
            p.file.write(qPrintable(QString("%1 <- summarySE(%1, measurevar=\"Y\", groupvars=c(\"%2\", \"X\"), conf.interval=confidence)"
                                            "\n").arg(data, groupvar)));
        }
        p.file.write(qPrintable(QString("%1 <- summarySE(%1, measurevar=\"X\", groupvars=c(\"Error\", \"%2\", \"Y\"), conf.interval=confidence)"
                                        "\n\n").arg("ERR", groupvar)));
    }

    // Use a br::file for simple storage of plot options
    QMap<QString,File> optMap;
    optMap.insert("rocOptions", File(QString("[xTitle=False Accept Rate,yTitle=True Accept Rate,xLog=true,yLog=false,xLimits=(.0000001,.1)]")));
    optMap.insert("detOptions", File(QString("[xTitle=False Accept Rate,yTitle=False Reject Rate,xLog=true,yLog=true,xLimits=(.0000001,.1),yLimits=(.0001,1)]")));
    optMap.insert("farOptions", File(QString("[xTitle=Score,yTitle=False Accept Rate,xLog=false,yLog=true,xLabels=waiver(),yLimits=(.0000001,1)]")));
    optMap.insert("frrOptions", File(QString("[xTitle=Score,yTitle=False Reject Rate,xLog=false,yLog=true,xLabels=waiver(),yLimits=(.0001,1)]")));

    foreach (const QString &key, optMap.keys()) {
        const QStringList options = destination.get<QStringList>(key, QStringList());
        foreach (const QString &option, options) {
            QStringList words = QtUtils::parse(option, '=');
            QtUtils::checkArgsSize(words[0], words, 1, 2);
            optMap[key].set(words[0], words[1]);
        }
    }

    // Write plots
    QString plot = "plot <- plotLine(lineData=%1, options=list(%2), flipY=%3)\nplot\n";
    p.file.write(qPrintable(QString(plot).arg("DET", toRList(optMap["rocOptions"]), "TRUE")));
    p.file.write(qPrintable(QString(plot).arg("DET", toRList(optMap["detOptions"]), "FALSE")));
    p.file.write("plot <- plotSD(sdData=SD)\nplot\n");
    p.file.write("plot <- plotBC(bcData=BC)\nplot\n");
    p.file.write(qPrintable(QString(plot).arg("FAR", toRList(optMap["farOptions"]), "FALSE")));
    p.file.write(qPrintable(QString(plot).arg("FRR", toRList(optMap["frrOptions"]), "FALSE")));

    return p.finalize(show);
}

} // namespace br
