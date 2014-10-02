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

static QStringList getPivots(const QString &file, bool headers)
{
    QString str;
    if (headers) str = QFileInfo(file).dir().dirName();
    else         str = QFileInfo(file).completeBaseName();
    return str.split("_");
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

    RPlot(QStringList files, const File &destination, bool isEvalFormat = true)
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
        // Format data
        if (isEvalFormat)
            file.write(qPrintable(QString("\n"
                       "# Split data into individual plots\n"
                       "plot_index = which(names(data)==\"Plot\")\n"
                       "Metadata <- data[grep(\"Metadata\",data$Plot),-c(1)]\n"
                       "TI <- data[grep(\"TI\",data$Plot),-c(1)]\n"
                       "BG <- data[grep(\"BG\",data$Plot),-c(1)]\n"
                       "DET <- data[grep(\"DET\",data$Plot),-c(1)]\n"
                       "FAR <- data[grep(\"FAR\",data$Plot),-c(1)]\n"
                       "FRR <- data[grep(\"FRR\",data$Plot),-c(1)]\n"
                       "SD <- data[grep(\"SD\",data$Plot),-c(1)]\n"
                       "FT <- data[grep(\"FT\",data$Plot),-c(1)]\n"
                       "CT <- data[grep(\"CT\",data$Plot),-c(1)]\n"
                       "BC <- data[grep(\"BC\",data$Plot),-c(1)]\n"
                       "CMC <- data[grep(\"CMC\",data$Plot),-c(1)]\n"
                       "FAR$Error <- \"FAR\"\n"
                       "FRR$Error <- \"FRR\"\n"
                       "ERR <- rbind(FAR, FRR)\n"
                       "rm(data, FAR, FRR)\n"
                       "\n"
                       "# Format data\n"
                       "Metadata$Y<-factor(Metadata$Y, levels=c(\"Genuine\",\"Impostor\",\"Ignored\",\"Gallery\",\"Probe\"))\n"
                       "TI$Y <- as.character(TI$Y)\n"
                       "BG$Y <- as.character(BG$Y)\n"
                       "DET$Y <- as.numeric(as.character(DET$Y))\n"
                       "ERR$Y <- as.numeric(as.character(ERR$Y))\n"
                       "SD$Y <- as.factor(unique(as.character(SD$Y)))\n"
                       "FT$Y <- as.numeric(as.character(FT$Y))\n"
                       "CT$Y <- as.numeric(as.character(CT$Y))\n"
                       "BC$Y <- as.numeric(as.character(BC$Y))\n"
                       "CMC$Y <- as.numeric(as.character(CMC$Y))\n"
                       "\n"
                       "# Code to format FAR values\n"
                       "far_names <- list('0.001'=\"FAR = 0.1%\", '0.01'=\"FAR = 1%\")\n"
                       "far_labeller <- function(variable,value) { return(far_names[as.character(value)]) }\n"
                       "\n"
                       "# Code to format TAR@FAR table\n"
                       "algs <- unique(FT$%1)\n"
                       "algs <- algs[!duplicated(algs)]\n"
                       "mat <- matrix(FT$Y,nrow=6,ncol=length(algs),byrow=FALSE)\n"
                       "colnames(mat) <- algs \n"
                       "rownames(mat) <- c(\"FAR = 1e-06\", \"FAR = 1e-05\", \"FAR = 1e-04\", \"FAR = 1e-03\", \"FAR = 1e-02\", \"FAR = 1e-01\")\n"
                       "FTtable <- as.table(mat)\n"
                       "\n"
                       "# Code to format CMC Table\n"
                       "mat <- matrix(CT$Y,nrow=6,ncol=length(algs),byrow=FALSE)\n"
                       "colnames(mat) <- algs \n"
                       "rownames(mat) <- c(\" Rank 1\", \"Rank 5\", \"Rank 10\", \"Rank 20\", \"Rank 50\", \"Rank 100\")\n"
                       "CMCtable <- as.table(mat)\n").arg(major.header)));
                       
        // Open output device
        file.write(qPrintable(QString("\n"
                                      "# Open output device\n"
                                      "%1(\"%2.%1\")\n").arg(suffix, basename)));

        // Write metadata table
        if ((suffix == "pdf") && isEvalFormat) {
            file.write("\n"
                       "# Write metadata table\n");
            QString textplot = "MT <- as.data.frame(Metadata[c(1,2,3,4,5),])\n"
                               "par(mfrow=c(4,1))\n"
                               "plot.new()\n"
                               "print(title(paste(\"%1 - %2\",date(),sep=\"\\n\")))\n"
                               "mat <- matrix(MT$X[c(1,2)],ncol=2)\n"
                               "colnames(mat) <- c(\"Gallery\", \"Probe\")\n"
                               "imageTable <- as.table(mat)\n"
                               "print(textplot(imageTable,show.rownames=FALSE))\n"
                               "print(title(\"Images\"))\n"
                               "mat <- matrix(MT$X[c(3,4,5)],ncol=3)\n"
                               "colnames(mat) <- c(\"Genuine\", \"Impostor\", \"Ignored\")\n"
                               "matchTable <- as.table(mat)\n"
                               "print(textplot(matchTable,show.rownames=FALSE))\n"
                               "print(title(\"Matches\"))\n"
                               "plot.new()\n"
                               "print(title(\"Gallery * Probe = Genuine + Impostor + Ignored\"))\n"
                               "plot.new()\n"
                               "print(textplot(FTtable))\n"
                               "print(title(\"Table of True Accept Rates at various False Accept Rates\"))\n"
                               "print(textplot(CMCtable))\n"
                               "print(title(\"Table of retrieval rate at various ranks\"))\n";
            file.write(qPrintable(textplot.arg(PRODUCT_NAME, PRODUCT_VERSION)));
        }

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

    const bool minimalist = destination.getBool("minimalist");

    // Use a br::file for simple storage of plot options
    File cmcOpts;
    const QStringList cmcOptions = destination.get<QStringList>("cmcOptions", QStringList());
    foreach (const QString& option, cmcOptions) {
        QStringList words = QtUtils::parse(option, '=');
        QtUtils::checkArgsSize(words[0],words,1,2);
        cmcOpts.set(words[0],words[1]);
    }

    File rocOpts;
    const QStringList rocOptions = destination.get<QStringList>("rocOptions", QStringList());
    foreach (const QString& option, rocOptions) {
        QStringList words = QtUtils::parse(option, '=');
        QtUtils::checkArgsSize(words[0],words,1,2);
        rocOpts.set(words[0],words[1]);
    }


    RPlot p(files, destination);

    p.file.write(qPrintable(QString("qplot(X, 1-Y, data=DET%1, main=\"%2\"").arg((p.major.smooth || p.minor.smooth) ? ", geom=\"smooth\", method=loess, level=0.99" : ", geom=\"line\"", rocOpts.get<QString>("title",QString())) +
                            (p.major.size > 1 ? QString(", colour=factor(%1)").arg(p.major.header) : QString()) +
                            (p.minor.size > 1 ? QString(", linetype=factor(%1)").arg(p.minor.header) : QString()) +
                            QString(", xlab=\"False Accept Rate\", ylab=\"True Accept Rate\") + theme_minimal()") +
                            (p.major.size > 1 ? getScale("colour", "Algorithm", p.major.size) : QString()) +
                            (p.minor.size > 1 ? QString(" + scale_linetype_discrete(\"%1\")").arg(p.minor.header) : QString()) +
                            QString(" + scale_x_log10(labels=trans_format(\"log10\", math_format()))") +
                            (rocOpts.contains("yLimits") ? QString(" + scale_y_continuous(labels=percent) + coord_cartesian(ylim=%1)").arg("c"+QtUtils::toString(rocOpts.get<QPointF>("yLimits",QPointF()))) : QString(" + scale_y_continuous(labels=percent)")) +
                            QString(" + annotation_logticks(sides=\"b\")") +
                            QString(" + theme(legend.title = element_text(size = %1), plot.title = element_text(size = %1), axis.text = element_text(size = %1), axis.title.x = element_text(size = %1), axis.title.y = element_text(size = %1),"
                            " legend.position=%2, legend.background = element_rect(fill = 'white'), panel.grid.major = element_line(colour = \"gray\"), panel.grid.minor = element_line(colour = \"gray\", linetype = \"dashed\"), legend.text = element_text(size = %1))\n\n").arg(QString::number(rocOpts.get<float>("textSize",12)), rocOpts.contains("legendPosition") ? "c"+QtUtils::toString(rocOpts.get<QPointF>("legendPosition")) : "'bottom'")));

    p.file.write(qPrintable(QString("qplot(X, Y, data=DET%1").arg((p.major.smooth || p.minor.smooth) ? ", geom=\"smooth\", method=loess, level=0.99" : ", geom=\"line\"") +
                            (p.major.size > 1 ? QString(", colour=factor(%1)").arg(p.major.header) : QString()) +
                            (p.minor.size > 1 ? QString(", linetype=factor(%1)").arg(p.minor.header) : QString()) +
                            QString(", xlab=\"False Accept Rate\", ylab=\"False Reject Rate\") + geom_abline(alpha=0.5, colour=\"grey\", linetype=\"dashed\") + theme_minimal()") +
                            (p.major.size > 1 ? getScale("colour", "Algorithm", p.major.size) : QString()) +
                            (p.minor.size > 1 ? QString(" + scale_linetype_discrete(\"%1\")").arg(p.minor.header) : QString()) +
                            QString(" + scale_x_log10(labels=trans_format(\"log10\", math_format())) + scale_y_log10(labels=trans_format(\"log10\", math_format())) + annotation_logticks()\n\n")));

    p.file.write(qPrintable(QString("qplot(X, data=SD, geom=\"histogram\", fill=Y, position=\"identity\", alpha=I(1/2)") +
                            QString(", xlab=\"Score\", ylab=\"Frequency\"") +
                            QString(") + scale_fill_manual(\"Ground Truth\", values=c(\"blue\", \"red\")) + theme_minimal() + scale_x_continuous(minor_breaks=NULL) + scale_y_continuous(minor_breaks=NULL) + theme(axis.text.y=element_blank(), axis.ticks=element_blank(), axis.text.x=element_text(angle=-90, hjust=0))") +
                            (p.major.size > 1 ? (p.minor.size > 1 ? QString(" + facet_grid(%2 ~ %1, scales=\"free\")").arg((p.flip ? p.major.header : p.minor.header), (p.flip ? p.minor.header : p.major.header)) : QString(" + facet_wrap(~ %1, scales = \"free\")").arg(p.major.header)) : QString()) +
                            QString(" + theme(aspect.ratio=1)\n\n")));

    p.file.write(qPrintable(QString("ggplot(CMC, aes(x=X, y=Y%1%2)) + ggtitle(\"%3\") + xlab(\"Rank\") + ylab(\"Retrieval Rate\")").arg(p.major.size > 1 ? QString(" ,colour=factor(%1)").arg(p.major.header) : QString(), p.minor.size > 1 ? QString(", linetype=factor(%1)").arg(p.minor.header) : QString(), cmcOpts.get<QString>("title",QString())) +
                            QString(((p.major.smooth || p.minor.smooth) ? (minimalist ? " + stat_summary(geom=\"line\", fun.y=mean, size=%1)" : " + stat_summary(geom=\"line\", fun.y=min, aes(linetype=\"Min/Max\"), size=%1) + stat_summary(geom=\"line\", "
                            "fun.y=max, aes(linetype=\"Min/Max\"), size=%1) + stat_summary(geom=\"line\", fun.y=mean, aes(linetype=\"Mean\"), size=%1) + scale_linetype_manual(\"Legend\", values=c(\"Mean\"=1, \"Min/Max\"=2))") : " + geom_line(size=%1)")).arg(QString::number(cmcOpts.get<float>("thickness",1))) +
                            (minimalist ? "" : " + scale_x_log10(labels=c(1,5,10,50,100), breaks=c(1,5,10,50,100)) + annotation_logticks(sides=\"b\")") +
                            (p.major.size > 1 ? getScale("colour", "Algorithm", p.major.size) : QString()) +
                            (p.minor.size > 1 ? QString(" + scale_linetype_discrete(\"%1\")").arg(p.minor.header) : QString()) +
                            (cmcOpts.contains("yLimits") ? QString(" + scale_y_continuous(labels=percent) + coord_cartesian(ylim=%1)").arg("c"+QtUtils::toString(cmcOpts.get<QPointF>("yLimits",QPointF()))) : QString(" + scale_y_continuous(labels=percent)")) +
                            QString(" + theme_minimal() + theme(legend.title = element_text(size = %1), plot.title = element_text(size = %1), axis.text = element_text(size = %1), axis.title.x = element_text(size = %1), axis.title.y = element_text(size = %1),"
                            " legend.position=%2, legend.background = element_rect(fill = 'white'), panel.grid.major = element_line(colour = \"gray\"), panel.grid.minor = element_line(colour = \"gray\", linetype = \"dashed\"), legend.text = element_text(size = %1))\n\n").arg(QString::number(cmcOpts.get<float>("textSize",12)), cmcOpts.contains("legendPosition") ? "c"+QtUtils::toString(cmcOpts.get<QPointF>("legendPosition")) : "'bottom'")));

    p.file.write(qPrintable(QString("qplot(factor(%1)%2, data=BC, %3").arg(p.major.smooth ? (p.minor.header.isEmpty() ? "Algorithm" : p.minor.header) : p.major.header, (p.major.smooth || p.minor.smooth) ? ", Y" : "", (p.major.smooth || p.minor.smooth) ? "geom=\"boxplot\"" : "geom=\"bar\", position=\"dodge\", weight=Y") +
                            (p.major.size > 1 ? QString(", fill=factor(%1)").arg(p.major.header) : QString()) +
                            QString(", xlab=\"False Accept Rate\", ylab=\"True Accept Rate\") + theme_minimal()") +
                            (p.major.size > 1 ? getScale("fill", "Algorithm", p.major.size) : QString()) +
                            (p.minor.size > 1 ? QString(" + facet_grid(%2 ~ X)").arg(p.minor.header) : QString(" + facet_grid(. ~ X, labeller=far_labeller)")) +
                            QString(" + scale_y_continuous(labels=percent) + theme(legend.position=\"none\", axis.text.x=element_text(angle=-90, hjust=0))%1").arg((p.major.smooth || p.minor.smooth) ? "" : " + geom_text(data=BC, aes(label=Y, y=0.05))\n\n")));

    p.file.write(qPrintable(QString("qplot(X, Y, data=ERR%1, linetype=Error").arg((p.major.smooth || p.minor.smooth) ? ", geom=\"smooth\", method=loess, level=0.99" : ", geom=\"line\"") +
                            ((p.flip ? p.major.size : p.minor.size) > 1 ? QString(", colour=factor(%1)").arg(p.flip ? p.major.header : p.minor.header) : QString()) +
                            QString(", xlab=\"Score\", ylab=\"Error Rate\") + theme_minimal()") +
                            ((p.flip ? p.major.size : p.minor.size) > 1 ? getScale("colour", p.flip ? "Algorithm" : "Algorithm", p.flip ? p.major.size : p.minor.size) : QString()) +
                            QString(" + scale_y_log10(labels=percent) + annotation_logticks(sides=\"l\")") +
                            ((p.flip ? p.minor.size : p.major.size) > 1 ? QString(" + facet_wrap(~ %1, scales=\"free_x\")").arg(p.flip ? p.minor.header : p.major.header) : QString()) +
                            QString(" + theme(aspect.ratio=1)\n\n")));

    p.file.write(qPrintable(QString("if (nrow(TI) != 0) {\n\tlibrary(jpeg)\n\tlibrary(png)\n\tlibrary(tiff)\n\tlibrary(grid)\n\t") +
                            QString("multiplot <- function(..., plotlist=NULL, cols) {\n\t") +
                            QString("\trequire(grid)\n\n\t\t# Make a list from the ... arguments and plotlist\n\t\tplots <- c(list(...), plotlist)\n") + 
                            QString("\t\tnumPlots = length(plots)\n\n\t\t# Make the panel\n\t\tplotCols = cols\n\t\tplotRows = ceiling(numPlots/plotCols)\n\n") +
                            QString("\t\t# Set up the page\n\t\tgrid.newpage()\n\t\tpushViewport(viewport(layout = grid.layout(plotRows, plotCols)))\n\t\tvplayout <- function(x, y)\n\t\t\tviewport(layout.pos.row = x, layout.pos.col = y)\n\n") +
                            QString("\t\t# Make each plot, in the correct location\n\t\tfor (i in 1:numPlots) {\n\t\t\tcurRow = ceiling(i/plotCols)\n\t\t\tcurCol = (i-1) %% plotCols + 1\n\t\t\tprint(plots[[i]], vp = vplayout(curRow, curCol))\n\t\t}\n\t}\n\n")));
    
    p.file.write(qPrintable(QString("\t# Print top impostor matches\n\tfor (i in 1:nrow(TI)) {\n\t\tscore <- TI[i,1]\n\t\tfiles <- TI[i,2]\n\t\talg <- TI[i,3]\n\t\tfiles <- unlist(strsplit(files, \"[:]\"))\n\n\t\text1 <- unlist(strsplit(files[2], \"[.]\"))[2]\n\t\text2 <- unlist(strsplit(files[4], \"[.]\"))[2]\n\t\t") +
                            QString("if (ext1 == \"jpg\" || ext1 == \"JPEG\" || ext1 == \"jpeg\" || ext1 == \"JPG\") {\n\t\t\timg1 <- readJPEG(files[2])\n\t\t} else if (ext1 == \"PNG\" || ext1 == \"png\") {\n\t\t\timg1 <- readPNG(files[2])\n\t\t} else if (ext1 == \"TIFF\" || ext1 == \"tiff\" || ext1 == \"TIF\" || ext1 == \"tif\") {\n\t\t\timg1 <- readTIFF(files[2])\n\t\t} else {\n\t\t\tnext\n\t\t}\n\t\tif (ext2 == \"jpg\" || ext2 == \"JPEG\" || ext2 == \"jpeg\" || ext2 == \"JPG\") {\n\t\t\timg2 <- readJPEG(files[4])\n\t\t} ") +
                            QString("else if (ext2 == \"PNG\" || ext2 == \"png\") {\n\t\t\timg2 <- readPNG(files[4])\n\t\t} else if (ext2 == \"TIFF\" || ext2 == \"tiff\" || ext2 == \"TIF\" || ext2 == \"tif\") {\n\t\t\timg2 <- readTIFF(files[4])\n\t\t} else {\n\t\t\tnext\n\t\t}") +
                            QString("\n\t\tname1 <- files[1]\n\t\tname2 <- files[3]\n\n\t\tg1 <- rasterGrob(img1, interpolate=TRUE)\n\t\tg2 <- rasterGrob(img2, interpolate=TRUE)\n\n\t\t") +
                            QString("plot1 <- qplot(1:10, 1:10, geom=\"blank\") + annotation_custom(g1, xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) + theme(axis.line=element_blank(), axis.text.x=element_blank(), axis.text.y=element_blank(), axis.ticks=element_blank(), panel.background=element_blank()) + labs(title=alg) + ylab(unlist(strsplit(files[2], \"[/]\"))[length(unlist(strsplit(files[2], \"[/]\")))]) + xlab(name1)\n\t\t") +
                            QString("plot2 <- qplot(1:10, 1:10, geom=\"blank\") + annotation_custom(g2, xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) + theme(axis.line=element_blank(), axis.text.x=element_blank(), axis.text.y=element_blank(), axis.ticks=element_blank(), panel.background=element_blank()) + labs(title=paste(\"Impostor score =\", score)) + ylab(unlist(strsplit(files[4], \"[/]\"))[length(unlist(strsplit(files[4], \"[/]\")))]) + xlab(name2)\n\n\t\t") +
                            QString("multiplot(plot1, plot2, cols=2)\n\t}")));

    p.file.write(qPrintable(QString("\n\n\t# Print worst genuine matches\n\tfor (i in 1:nrow(BG)) {\n\t\tscore <- BG[i,1]\n\t\tfiles <- BG[i,2]\n\t\talg <- BG[i,3]\n\t\tfiles <- unlist(strsplit(files, \"[:]\"))\n\n\t\text1 <- unlist(strsplit(files[2], \"[.]\"))[2]\n\t\text2 <- unlist(strsplit(files[4], \"[.]\"))[2]\n\t\t") +
                            QString("if (ext1 == \"jpg\" || ext1 == \"JPEG\" || ext1 == \"jpeg\" || ext1 == \"JPG\") {\n\t\t\timg1 <- readJPEG(files[2])\n\t\t} else if (ext1 == \"PNG\" || ext1 == \"png\") {\n\t\t\timg1 <- readPNG(files[2])\n\t\t} else if (ext1 == \"TIFF\" || ext1 == \"tiff\" || ext1 == \"TIF\" || ext1 == \"tif\") {\n\t\t\timg1 <- readTIFF(files[2])\n\t\t} else {\n\t\t\tnext\n\t\t}\n\t\tif (ext2 == \"jpg\" || ext2 == \"JPEG\" || ext2 == \"jpeg\" || ext2 == \"JPG\") {\n\t\t\timg2 <- readJPEG(files[4])\n\t\t} ") +
                            QString("else if (ext2 == \"PNG\" || ext2 == \"png\") {\n\t\t\timg2 <- readPNG(files[4])\n\t\t} else if (ext2 == \"TIFF\" || ext2 == \"tiff\" || ext2 == \"TIF\" || ext2 == \"tif\") {\n\t\t\timg2 <- readTIFF(files[4])\n\t\t} else {\n\t\t\tnext\n\t\t}") +
                            QString("\n\t\tname1 <- files[1]\n\t\tname2 <- files[3]\n\n\t\tg1 <- rasterGrob(img1, interpolate=TRUE)\n\t\tg2 <- rasterGrob(img2, interpolate=TRUE)\n\n\t\t") +
                            QString("plot1 <- qplot(1:10, 1:10, geom=\"blank\") + annotation_custom(g1, xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) + theme(axis.line=element_blank(), axis.text.x=element_blank(), axis.text.y=element_blank(), axis.ticks=element_blank(), panel.background=element_blank()) + labs(title=alg) + ylab(unlist(strsplit(files[2], \"[/]\"))[length(unlist(strsplit(files[2], \"[/]\")))]) + xlab(name1)\n\t\t") +
                            QString("plot2 <- qplot(1:10, 1:10, geom=\"blank\") + annotation_custom(g2, xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) + theme(axis.line=element_blank(), axis.text.x=element_blank(), axis.text.y=element_blank(), axis.ticks=element_blank(), panel.background=element_blank()) + labs(title=paste(\"Genuine score =\", score)) + ylab(unlist(strsplit(files[4], \"[/]\"))[length(unlist(strsplit(files[4], \"[/]\")))]) + xlab(name2)\n\n\t\t") +
                            QString("multiplot(plot1, plot2, cols=2)\n\t}\n}\n\n")));

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
    RPlot p(files, destination, false);

    p.file.write("# Split data into individual plots\n"
                 "plot_index = which(names(data)==\"Plot\")\n"
                 "DiscreteROC <- data[grep(\"DiscreteROC\",data$Plot),-c(1)]\n"
                 "ContinuousROC <- data[grep(\"ContinuousROC\",data$Plot),-c(1)]\n"
                 "DiscretePR <- data[grep(\"DiscretePR\",data$Plot),-c(1)]\n"
                 "ContinuousPR <- data[grep(\"ContinuousPR\",data$Plot),-c(1)]\n"
                 "Overlap <- data[grep(\"Overlap\",data$Plot),-c(1)]\n"
                 "AverageOverlap <- data[grep(\"AverageOverlap\",data$Plot),-c(1)]\n"
                 "rm(data)\n"
                 "\n");

    QString plotType("line");
    if (filesHaveSinglePoint(files))
        plotType = QString("point");

    foreach (const QString &type, QStringList() << "Discrete" << "Continuous")
        p.file.write(qPrintable(QString("qplot(X, Y, data=%1ROC%2").arg(type, (p.major.smooth || p.minor.smooth) ? ", geom=\"smooth\", method=loess, level=0.99" : QString(", geom=\"%1\"").arg(plotType)) +
                                (p.major.size > 1 ? QString(", colour=factor(%1)").arg(p.major.header) : QString()) +
                                (p.minor.size > 1 ? QString(", linetype=factor(%1)").arg(p.minor.header) : QString()) +
                                QString(", xlab=\"False Accepts Per Image\", ylab=\"True Accept Rate\") + theme_minimal()") +
                                (p.major.size > 1 ? getScale("colour", p.major.header, p.major.size) : QString()) +
                                (p.minor.size > 1 ? QString(" + scale_linetype_discrete(\"%1\")").arg(p.minor.header) : QString()) +
                                QString(" + scale_x_log10() + scale_y_continuous(labels=percent, limits=c(0,1)) + annotation_logticks(sides=\"b\") + ggtitle(\"%1\") + theme(legend.position=\"bottom\")\n\n").arg(type)));

    foreach (const QString &type, QStringList() << "Discrete" << "Continuous")
        p.file.write(qPrintable(QString("qplot(X, Y, data=%1PR%2").arg(type, (p.major.smooth || p.minor.smooth) ? ", geom=\"smooth\", method=loess, level=0.99" : QString(", geom=\"%1\"").arg(plotType)) +
                                (p.major.size > 1 ? QString(", colour=factor(%1)").arg(p.major.header) : QString()) +
                                (p.minor.size > 1 ? QString(", linetype=factor(%1)").arg(p.minor.header) : QString()) +
                                QString(", xlab=\"Recall\", ylab=\"Precision\") + theme_minimal()") +
                                (p.major.size > 1 ? getScale("colour", p.major.header, p.major.size) : QString()) +
                                (p.minor.size > 1 ? QString(" + scale_linetype_discrete(\"%1\")").arg(p.minor.header) : QString()) +
                                QString(" + scale_x_continuous(limits=c(0,1)) + scale_y_continuous(labels=percent, limits=c(0,1)) + ggtitle(\"%1\") + theme(legend.position=\"bottom\")\n\n").arg(type)));

    p.file.write(qPrintable(QString("qplot(X, data=Overlap, geom=\"histogram\", position=\"identity\", xlab=\"Overlap\", ylab=\"Frequency\")") +
                            QString(" + theme_minimal() + scale_x_continuous(minor_breaks=NULL) + scale_y_continuous(minor_breaks=NULL) + theme(axis.text.y=element_blank(), axis.ticks=element_blank(), axis.text.x=element_text(angle=-90, hjust=0))") +
                            (p.major.size > 1 ? (p.minor.size > 1 ? QString(" + facet_grid(%2 ~ %1, scales=\"free\")").arg(p.minor.header, p.major.header) : QString(" + facet_wrap(~ %1, scales = \"free\")").arg(p.major.header)) : QString()) +
                            QString(" + theme(aspect.ratio=1, legend.position=\"bottom\")\n\n")));

    p.file.write(qPrintable(QString("ggplot(AverageOverlap, aes(x=%1, y=%2, label=round(X,3)), main=\"Average Overlap\") + geom_text() + theme_minimal()").arg(p.minor.size > 1 ? p.minor.header : "'X'", p.major.size > 1 ? p.major.header : "'Y'") +
                            QString("%1%2\n\n").arg(p.minor.size > 1 ? "" : " + xlab(NULL)", p.major.size > 1 ? "" : " + ylab(NULL)")));

    p.file.write(qPrintable(QString("ggplot(AverageOverlap, aes(x=%1, y=%2, fill=X)) + geom_tile() + scale_fill_continuous(\"Average Overlap\") + theme_minimal()").arg(p.minor.size > 1 ? p.minor.header : "'X'", p.major.size > 1 ? p.major.header : "'Y'") +
                            QString("%1%2\n\n").arg(p.minor.size > 1 ? "" : " + xlab(NULL)", p.major.size > 1 ? "" : " + ylab(NULL)")));

    return p.finalize(show);
}

bool PlotLandmarking(const QStringList &files, const File &destination, bool show)
{
    qDebug("Plotting %d landmarking file(s) to %s", files.size(), qPrintable(destination));
    RPlot p(files, destination, false);

    p.file.write("# Split data into individual plots\n"
                 "plot_index = which(names(data)==\"Plot\")\n"
                 "Box <- data[grep(\"Box\",data$Plot),-c(1)]\n"
                 "rm(data)\n"
                 "\n");

    p.file.write(qPrintable(QString("ggplot(Box, aes(Y,%1%2))").arg(p.major.size > 1 ? QString(", colour=%1").arg(p.major.header) : QString(), p.minor.size > 1 ? QString(", linetype=%1").arg(p.minor.header) : QString()) +
                            QString(" + annotation_logticks(sides=\"b\") + stat_ecdf() + scale_x_log10(\"Normalized Error\", breaks=c(0.001,0.01,0.1,1,10)) + scale_y_continuous(\"Cumulative Density\", label=percent) + theme_minimal()\n\n")));
    p.file.write(qPrintable(QString("ggplot(Box, aes(factor(X), Y%1%2))").arg(p.major.size > 1 ? QString(", colour=%1").arg(p.major.header) : QString(), p.minor.size > 1 ? QString(", linetype=%1").arg(p.minor.header) : QString()) +
                            QString("+ annotation_logticks(sides=\"l\") + geom_boxplot(alpha=0.5) + geom_jitter(size=1, alpha=0.5) + scale_x_discrete(\"Landmark\") + scale_y_log10(\"Normalized Error\", breaks=c(0.01,0.1,1,10)) + theme_minimal()\n\n")));
    p.file.write(qPrintable(QString("ggplot(Box, aes(factor(X), Y%1%2))").arg(p.major.size > 1 ? QString(", colour=%1").arg(p.major.header) : QString(), p.minor.size > 1 ? QString(", linetype=%1").arg(p.minor.header) : QString()) +
                            QString("+ annotation_logticks(sides=\"l\") + geom_violin(alpha=0.5) + scale_x_discrete(\"Landmark\") + scale_y_log10(\"Normalized Error\", breaks=c(0.001,0.01,0.1,1,10))\n\n")));

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
