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

const QString plottingUtils = QString(
"# Load libraries\n"
"library(\"ggplot2\")\n"
"library(\"gplots\")\n"
"library(\"reshape\")\n"
"library(\"scales\")\n"
"library(\"jpeg\")\n"
"library(\"png\")\n"
"library(\"grid\")\n"
"\n"
"# Code to format FAR values\n"
"far_names <- list('0.0001'=\"FAR = 0.01%\", '0.001'=\"FAR = 0.1%\", '0.01'=\"FAR = 1%\")\n"
"far_labeller <- function(variable,value) {\n"
"    if (as.character(value) %in% names(far_names)) {\n"
"        return(far_names[as.character(value)])\n"
"    } else {\n"
"        return(as.character(value))\n"
"    }\n"
"}\n"
"\n"
"getScale <- function(mode, title, vals) {\n"
"    if      (vals > 12) return(do.call(paste(\"scale\", mode, \"discrete\", sep=\"_\"), list(title)))\n"
"    else if (vals > 11) return(do.call(paste(\"scale\", mode, \"brewer\", sep=\"_\"), list(title, palette=(if(seq) \"Reds\" else \"Set3\"), type=(if(seq) \"seq\" else \"qual\"))))\n"
"    else if (vals > 9)  return(do.call(paste(\"scale\", mode, \"brewer\", sep=\"_\"), list(title, palette=(if(seq) \"Reds\" else \"Paired\"), type=(if(seq) \"seq\" else \"qual\"))))\n"
"    else                return(do.call(paste(\"scale\", mode, \"brewer\", sep=\"_\"), list(title, palette=(if(seq) \"Reds\" else \"Set1\"), type=(if(seq) \"seq\" else \"qual\"))))\n"
"}\n"
"\n"
"plotMetadata <- function(metadata=NULL, title=NULL) {\n"
"    MT <- as.data.frame(metadata[c(1, 2, 3, 4, 5),])\n"
"    par(mfrow=c(4, 1))\n"
"    plot.new()\n"
"    print(title(paste(title, date(), sep=\"\n\")))\n"
"    mat <- matrix(MT$X[c(1, 2)], ncol=2)\n"
"    colnames(mat) <- c(\"Gallery\", \"Probe\")\n"
"    imageTable <- as.table(mat)\n"
"    print(textplot(imageTable, show.rownames=FALSE))\n"
"    print(title(\"Images\"))\n"
"    mat <- matrix(MT$X[c(3, 4, 5)], ncol=3)\n"
"    colnames(mat) <- c(\"Genuine\", \"Impostor\", \"Ignored\")\n"
"    matchTable <- as.table(mat)\n"
"    print(textplot(matchTable, show.rownames=FALSE))\n"
"    print(title(\"Matches\"))\n"
"    plot.new()\n"
"    print(title(\"Gallery * Probe = Genuine + Impostor + Ignored\"))\n"
"}\n"
"\n"
"plotTable <- function(tableData=NULL, name=NULL, labels=NULL) {\n"
"    if (nrow(tableData) == 0) return()\n"
"    if (smooth && confidence != 0) {\n"
"        input = paste(as.character(round(tableData$Y, 3)), round(tableData$ci, 3), sep=\"\u00b1\")\n"
"    } else {\n"
"        input = tableData$Y\n"
"    }\n"
"    mat <- matrix(input, nrow=length(labels), ncol=length(algs), byrow=FALSE)\n"
"    colnames(mat) <- algs\n"
"    rownames(mat) <- labels\n"
"    table <- as.table(mat)\n"
"    if (csv) {\n"
"        write.csv(table, file=paste(paste(basename, deparse(substitute(data)), sep=\"_\"), \".csv\", sep=\"\"))\n"
"    } else {\n"
"        print(textplot(table))\n"
"        print(title(name))\n"
"    }\n"
"}\n"
"\n"
"plotFRR <- function(tableData=NULL, operatingPoint=1e-4) {\n"
"    if (nrow(tableData) == 0) return()\n"
"    major <- majorHeader\n"
"    minor <- if(minorHeader == \"\") majorHeader else minorHeader\n"
"    tableData <- tableData[grep(operatingPoint,tableData$X),]\n"
"    mat <- matrix(tableData$Y, nrow=length(tableData[,major][!duplicated(tableData[,major])]), ncol=length(tableData[,minor][!duplicated(tableData[,minor])]), byrow=FALSE)\n"
"    colnames(mat) <- tableData[,minor][!duplicated(tableData[,minor])]\n"
"    rownames(mat) <- tableData[,major][!duplicated(tableData[,major])]\n"
"    table <- as.table(mat)\n"
"    options(scipen = 999)\n"
"    print(textplot(table))\n"
"    options(scipen = 0)\n"
"    print(title(paste(\"False Reject Rate at FAR=\", toString(operatingPoint))))\n"
"}\n"
"\n"
"plotLandmarkTables <- function(tableData=NULL) {\n"
"    if (majorSize > 1) {\n"
"        var <- majorHeader\n"
"    } else {\n"
"        if (minorHeader == \"\") var <- majorHeader else var <- minorHeader\n"
"    }\n"
"    StatBox <- summarySE(tableData, measurevar=\"Y\", groupvars=c(var,\"X\"))\n"
"    OverallStatBox <- summarySE(tableData, measurevar=\"Y\", groupvars=c(var))\n"
"    mat <- matrix(paste(as.character(round(StatBox$Y, 3)), round(StatBox$ci, 3), sep=\" \u00b1 \"), nrow=rows, ncol=length(algs), byrow=FALSE)\n"
"    mat <- rbind(mat, paste(as.character(round(OverallStatBox$Y, 3)), round(OverallStatBox$ci, 3), sep=\" \u00b1 \"))\n"
"    mat <- rbind(mat, as.character(round(NormLength$Y, 3)))\n"
"    colnames(mat) <- algs\n"
"    rownames(mat) <- c(seq(0, rows-1), \"Aggregate\",\"Average IPD\")\n"
"    ETable <- as.table(mat)\n"
"    print(textplot(ETable))\n"
"    print(title(\"Landmarking Error Rates\"))\n"
"}\n"
"\n"
"plotLine <- function(lineData=NULL, options=NULL, flipY=FALSE, geometry=\"path\") {\n"
"    textSize <- if(\"textSize\" %in% names(options)) as.numeric(options$textSize) else 12\n"
"    p <- qplot(X, if(flipY) 1-Y else Y, data=lineData, main=options$title, geom=geometry, size=if(\"size\" %in% names(options)) I(as.numeric(options$size)) else I(.5), colour=if(majorSize > 1) factor(eval(parse(text=majorHeader))) else NULL, linetype=if(minorSize > 1) factor(eval(parse(text=minorHeader))) else NULL, xlab=options$xTitle, ylab=options$yTitle) + theme_minimal()\n"
"    if (smooth && deparse(substitute(lineData)) != \"CMC\" && confidence != 0) p <- p + geom_errorbar(data=lineData[seq(1, NROW(lineData), by = 29),], aes(x=X, ymin=if(flipY) (1-lower) else lower, ymax=if(flipY) (1-upper) else upper), width=0.1, alpha=I(1/2))\n"
"    if (majorSize > 1) p <- p + getScale(\"colour\", majorHeader, majorSize)\n"
"    if (minorSize > 1) p <- p + scale_linetype_discrete(minorHeader)\n"
"\n"
"    # Set log/continuous scales, breaks and labels\n"
"    if (options$xLog)\n"
"        p <- p + scale_x_log10(labels=if(\"xLabels\" %in% names(options)) eval(parse(text=options$xLabels)) else trans_format(\"log10\", math_format()), breaks=if(\"xBreaks\" %in% names(options)) eval(parse(text=options$xBreaks)) else waiver()) + annotation_logticks(sides=\"b\")\n"
"    else\n"
"        p <- p + scale_x_continuous(labels=if(\"xLabels\" %in% names(options)) eval(parse(text=options$xLabels)) else percent, breaks=if(\"xBreaks\" %in% names(options)) eval(parse(text=options$xBreaks)) else pretty_breaks(n=10))\n"
"    if (options$yLog)\n"
"        p <- p + scale_y_log10(labels=if(\"yLabels\" %in% names(options)) eval(parse(text=options$yLabels)) else trans_format(\"log10\", math_format()), breaks=if(\"yBreaks\" %in% names(options)) eval(parse(text=options$yBreaks)) else waiver()) + annotation_logticks(sides=\"l\")\n"
"    else\n"
"        p <- p + scale_y_continuous(labels=if(\"yLabels\" %in% names(options)) eval(parse(text=options$yLabels)) else percent, breaks=if(\"yBreaks\" %in% names(options)) eval(parse(text=options$yBreaks)) else pretty_breaks(n=10))\n"
"\n"
"    if (\"xLimits\" %in% names(options) && \"yLimits\" %in% names(options)) {\n"
"        p <- p + coord_cartesian(xlim=eval(parse(text=options$xLimits)), ylim=eval(parse(text=options$yLimits)))\n"
"    } else {\n"
"        if (\"xLimits\" %in% names(options)) p <- p + coord_cartesian(xlim=eval(parse(text=options$xLimits)))\n"
"        if (\"yLimits\" %in% names(options)) p <- p + coord_cartesian(ylim=eval(parse(text=options$yLimits)))\n"
"    }\n"
"    p <- p + theme(legend.title = element_text(size = textSize), legend.text = element_text(size = textSize), plot.title = element_text(size = textSize), axis.text = element_text(size = textSize), axis.title.x = element_text(size = textSize), axis.title.y = element_text(size = textSize), legend.position=if(\"legendPosition\" %in% names(options)) eval(parse(text=options$legendPosition)) else \"bottom\", legend.background = element_rect(fill = 'white'), panel.grid.major = element_line(colour = \"gray\"), panel.grid.minor = element_line(colour = \"gray\", linetype = \"dashed\"))\n"
"    p <- p + guides(colour=guide_legend(ncol=ncol)) + guides(linetype=guide_legend(ncol=ncol))\n"
"    return(p)\n"
"}\n"
"\n"
"plotSD <- function(sdData=NULL) {\n"
"    p <- qplot(X, data=sdData, geom=\"histogram\", fill=Y, position=\"identity\", alpha=I(1/2), xlab=\"Score\", ylab=\"Frequency\")\n"
"    p <- p + scale_fill_manual(\"Ground Truth\", values=c(\"blue\", \"red\")) + theme_minimal() + scale_x_continuous(minor_breaks=NULL) + scale_y_continuous(minor_breaks=NULL) + theme(axis.text.y=element_blank(), axis.ticks=element_blank(), axis.text.x=element_text(angle=-90, hjust=0))\n"
"    if (majorSize > 1) {\n"
"        if (minorSize > 1) {\n"
"            if (flip) {\n"
"                A <- minorHeader\n"
"                B <- majorHeader\n"
"            } else {\n"
"                A <- majorHeader\n"
"                B <- minorHeader\n"
"            }\n"
"            p <- p + facet_grid(facets=as.formula(paste(A, \"~\", B)), scales=\"free\")\n"
"        } else {\n"
"            p <- p + facet_wrap(facets=as.formula(paste(\"~\", majorHeader)), scales=\"free\")\n"
"        }\n"
"    }\n"
"    p <- p + theme(aspect.ratio=1)\n"
"    return(p)\n"
"}\n"
"\n"
"plotBC <- function(bcData=NULL) {\n"
"    factor <- if (majorSmooth) minorHeader else majorHeader\n"
"    plotString <- paste(\"qplot(factor(\", factor, \")\", if(smooth) \", Y\" else \"\", \", data=bcData, \", if(smooth) \"geom=\\\"boxplot\\\"\" else \"geom=\\\"bar\\\", position=\\\"dodge\\\", weight=Y\", sep=\"\")\n"
"    p <- eval(parse(text=paste(plotString, if(majorSize > 1) paste(\", fill=factor(\", majorHeader, \")\", sep=\"\") else \"\", \", xlab=\\\"False Accept Rate\\\", ylab=\\\"True Accept Rate\\\") + theme_minimal()\", sep=\"\")))\n"
"    if (majorSize > 1) p <- p + getScale(\"fill\", majorHeader, majorSize)\n"
"    if (minorSize > 1) p <- p + facet_grid(facets=as.formula(paste(minorHeader, \"~\", \"X\")), labeller=far_labeller) else p <- p + facet_grid(. ~ X, labeller=far_labeller)\n"
"    p <- p + scale_y_continuous(labels=percent) + theme(legend.position=\"none\", axis.text.x=element_text(angle=-90, hjust=0))\n"
"    if (!smooth) p <- p + geom_text(data=bcData, aes(label=Y, y=0.05))\n"
"    return(p)\n"
"}\n"
"\n"
"plotOverlap <- function(overlapData=NULL) {\n"
"    p <- qplot(X, data=overlapData, geom=\"histogram\", position=\"identity\", xlab=\"Overlap\", ylab=\"Frequency\")\n"
"    p <- p + theme_minimal() + scale_x_continuous(minor_breaks=NULL) + scale_y_continuous(minor_breaks=NULL) + theme(axis.text.y=element_blank(), axis.ticks=element_blank(), axis.text.x=element_text(angle=-90, hjust=0))\n"
"    if (majorSize > 1) {\n"
"        if (minorSize > 1) {\n"
"            p <- p + facet_grid(facets=as.formula(paste(minorHeader, \"~\", majorHeader)), scales=\"free\")\n"
"        } else {\n"
"            p <- p + facet_wrap(facets=as.formula(paste(\"~\", majorHeader)), scales=\"free\")\n"
"        }\n"
"    }\n"
"    p <- p + theme(aspect.ratio=1, legend.position=\"bottom\")\n"
"    return(p)\n"
"}\n"
"\n"
"formatData <- function(type=\"eval\") {\n"
"    if (type == \"eval\") {\n"
"	    # Split data into individual plots\n"
"        plot_index <<- which(names(data)==\"Plot\")\n"
"        Metadata <<- data[grep(\"Metadata\",data$Plot),-c(1)]\n"
"        IM <<- data[grep(\"IM\",data$Plot),-c(1)]\n"
"        GM <<- data[grep(\"GM\",data$Plot),-c(1)]\n"
"        DET <<- data[grep(\"DET\",data$Plot),-c(1)]\n"
"        IET <<- data[grep(\"IET\",data$Plot),-c(1)]\n"
"        FAR <<- data[grep(\"FAR\",data$Plot),-c(1)]\n"
"        FRR <<- data[grep(\"FRR\",data$Plot),-c(1)]\n"
"        SD <<- data[grep(\"SD\",data$Plot),-c(1)]\n"
"        FF <<- data[grep(\"FF\",data$Plot),-c(1)]\n"
"        FT <<- data[grep(\"FT\",data$Plot),-c(1)]\n"
"        CT <<- data[grep(\"CT\",data$Plot),-c(1)]\n"
"        BC <<- data[grep(\"BC\",data$Plot),-c(1)]\n"
"        TS <<- data[grep(\"TS\",data$Plot),-c(1)]\n"
"        CMC <<- data[grep(\"CMC\",data$Plot),-c(1)]\n"
"\n"
"        # Format data\n"
"        Metadata$Y<-factor(Metadata$Y, levels=c(\"Genuine\", \"Impostor\", \"Ignored\", \"Gallery\", \"Probe\"))\n"
"        IM$Y <<- as.character(IM$Y)\n"
"        GM$Y <<- as.character(GM$Y)\n"
"        DET$Y <<- as.numeric(as.character(DET$Y))\n"
"        IET$Y <<- as.numeric(as.character(IET$Y))\n"
"        FAR$Y <<- as.numeric(as.character(FAR$Y))\n"
"        FRR$Y <<- as.numeric(as.character(FRR$Y))\n"
"        SD$Y <<- as.factor(unique(as.character(SD$Y)))\n"
"        FF$Y <<- as.numeric(as.character(FF$Y))\n"
"        FT$Y <<- as.numeric(as.character(FT$Y))\n"
"        CT$Y <<- as.numeric(as.character(CT$Y))\n"
"        BC$Y <<- as.numeric(as.character(BC$Y))\n"
"        TS$Y <<- as.character(TS$Y)\n"
"        CMC$Y <<- as.numeric(as.character(CMC$Y))\n"
"    } else if (type == \"detection\") {\n"
"        # Split data into individual plots\n"
"        DiscreteROC <<- data[grep(\"DiscreteROC\",data$Plot),-c(1)]\n"
"        ContinuousROC <<- data[grep(\"ContinuousROC\",data$Plot),-c(1)]\n"
"        DiscretePR <<- data[grep(\"DiscretePR\",data$Plot),-c(1)]\n"
"        ContinuousPR <<- data[grep(\"ContinuousPR\",data$Plot),-c(1)]\n"
"        Overlap <<- data[grep(\"Overlap\",data$Plot),-c(1)]\n"
"        AverageOverlap <<- data[grep(\"AverageOverlap\",data$Plot),-c(1)]\n"
"    } else if (type == \"landmarking\") {\n"
"        # Split data into individual plots\n"
"        Box <<- data[grep(\"Box\",data$Plot),-c(1)]\n"
"        Box$X <<- factor(Box$X, levels = unique(Box$X), ordered = TRUE)\n"
"        Sample <<- data[grep(\"Sample\",data$Plot),-c(1)]\n"
"        Sample$X <<- as.character(Sample$X)\n"
"        displaySample <<- readImageData(Sample)\n"
"        rows <<- displaySample[[1]]$value\n"
"        EXT <<- data[grep(\"EXT\",data$Plot),-c(1)]\n"
"        EXT$X <<- as.character(EXT$X)\n"
"        EXP <<- data[grep(\"EXP\",data$Plot),-c(1)]\n"
"        EXP$X <<- as.character(EXP$X)\n"
"        NormLength <<- data[grep(\"NormLength\",data$Plot),-c(1)]\n"
"    } else if (type == \"knn\") {\n"
"        # Split data into individual plots\n"
"        IET <<- data[grep(\"IET\",data$Plot),-c(1)]\n"
"        IET$Y <<- as.numeric(as.character(IET$Y))\n"
"        CMC <<- data[grep(\"CMC\",data$Plot),-c(1)]\n"
"        CMC$Y <<- as.numeric(as.character(CMC$Y))\n"
"    }\n"
"}\n"
"\n"
"summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE, conf.interval=0.95, .drop=TRUE) {\n"
"    # derived from http://www.cookbook-r.com/Manipulating_data/Summarizing_data/\n"
"	require(plyr)\n"
"\n"
"	length2 <- function (x, na.rm=FALSE) {\n"
"		if (na.rm) sum(!is.na(x))\n"
"		else       length(x)\n"
"	}\n"
"\n"
"	datac <- ddply(data, groupvars, .drop=.drop, .fun = function(xx, col) {\n"
"		c(N=length2(xx[[col]], na.rm=na.rm), mean=mean(xx[[col]], na.rm=na.rm), sd=sd(xx[[col]], na.rm=na.rm))\n"
"		},\n"
"		measurevar\n"
"	)\n"
"\n"
"	datac <- rename(datac, c(\"mean\" = measurevar))\n"
"	datac$se <- datac$sd / sqrt(datac$N)\n"
"	ciMult <- qt(conf.interval/2 + .5, datac$N-1)\n"
"	datac$ci <- datac$se * ciMult\n"
"\n"
"	datac$upper <- ifelse(datac[,measurevar] + datac$ci < 1, datac[,measurevar] + datac$ci, 1)\n"
"	datac$lower <- ifelse(datac[,measurevar] - datac$ci > 0, datac[,measurevar] - datac$ci, 0)\n"
"\n"
"	return(datac)\n"
"}\n"
"\n"
"multiplot <- function(..., plotlist=NULL, cols) {\n"
"    require(grid)\n"
"    # Make a list from the ... arguments and plotlist\n"
"    plots <- c(list(...), plotlist)\n"
"    numPlots = length(plots)\n"
"    # Make the panel\n"
"    plotCols = cols\n"
"    plotRows = ceiling(numPlots/plotCols)\n"
"    # Set up the page\n"
"    grid.newpage()\n"
"    pushViewport(viewport(layout = grid.layout(plotRows, plotCols)))\n"
"    vplayout <- function(x, y)\n"
"    viewport(layout.pos.row = x, layout.pos.col = y)\n"
"    # Make each plot, in the correct location\n"
"    for (i in 1:numPlots) {\n"
"        curRow = ceiling(i/plotCols)\n"
"        curCol = (i-1) %% plotCols + 1\n"
"        print(plots[[i]], vp = vplayout(curRow, curCol))\n"
"    }\n"
"}\n"
"\n"
"plotEERSamples <- function(imData=NULL, gmData=NULL) {\n"
"    if(nrow(imData) == 0) return()\n"
"\n"
"    printImages <- function(images, label) {\n"
"        for (i in 1:nrow(images)) {\n"
"            score <- images[i,1]\n"
"            files <- images[i,2]\n"
"            alg <- images[i,3]\n"
"            files <- unlist(strsplit(files, \"[:]\"))\n"
"\n"
"            ext1 <- unlist(strsplit(files[2], \"[.]\"))[2]\n"
"            ext2 <- unlist(strsplit(files[4], \"[.]\"))[2]\n"
"            if (ext1 == \"jpg\" || ext1 == \"JPEG\" || ext1 == \"jpeg\" || ext1 == \"JPG\") {\n"
"                img1 <- readJPEG(files[2])\n"
"            } else if (ext1 == \"PNG\" || ext1 == \"png\") {\n"
"                img1 <- readPNG(files[2])\n"
"            } else if (ext1 == \"TIFF\" || ext1 == \"tiff\" || ext1 == \"TIF\" || ext1 == \"tif\") {\n"
"                img1 <- readTIFF(files[2])\n"
"            } else {\n"
"                next\n"
"            }\n"
"            if (ext2 == \"jpg\" || ext2 == \"JPEG\" || ext2 == \"jpeg\" || ext2 == \"JPG\") {\n"
"                img2 <- readJPEG(files[4])\n"
"            } else if (ext2 == \"PNG\" || ext2 == \"png\") {\n"
"                img2 <- readPNG(files[4])\n"
"            } else if (ext2 == \"TIFF\" || ext2 == \"tiff\" || ext2 == \"TIF\" || ext2 == \"tif\") {\n"
"                img2 <- readTIFF(files[4])\n"
"            } else {\n"
"                next\n"
"            }\n"
"            name1 <- files[1]\n"
"            name2 <- files[3]\n"
"\n"
"            g1 <- rasterGrob(img1, interpolate=TRUE)\n"
"            g2 <- rasterGrob(img2, interpolate=TRUE)\n"
"\n"
"            plot1 <- qplot(1:10, 1:10, geom=\"blank\") + annotation_custom(g1, xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) + theme(axis.line=element_blank(), axis.text.x=element_blank(), axis.text.y=element_blank(), axis.ticks=element_blank(), panel.background=element_blank()) + labs(title=alg) + ylab(unlist(strsplit(files[2], \"[/]\"))[length(unlist(strsplit(files[2], \"[/]\")))]) + xlab(name1)\n"
"            plot2 <- qplot(1:10, 1:10, geom=\"blank\") + annotation_custom(g2, xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) + theme(axis.line=element_blank(), axis.text.x=element_blank(), axis.text.y=element_blank(), axis.ticks=element_blank(), panel.background=element_blank()) + labs(title=paste(label, \" score =\", score)) + ylab(unlist(strsplit(files[4], \"[/]\"))[length(unlist(strsplit(files[4], \"[/]\")))]) + xlab(name2)\n"
"\n"
"            multiplot(plot1, plot2, cols=2)\n"
"        }\n"
"    }\n"
"    printImages(imData, \"Impostor\")\n"
"    printImages(gmData, \"Genuine\")\n"
"}\n"
"\n"
"plotLandmarkSamples <- function(displaySample=NULL, expData=NULL, extData=NULL) {\n"
"    print(plotImage(displaySample[[1]], \"Sample Landmarks\", sprintf(\"Total Landmarks: %s\", displaySample[[1]]$value)))\n"
"    column <- if(majorSize > 1) majorHeader else minorHeader\n"
"    if (nrow(EXT) != 0 && nrow(EXP)) {\n"
"        for (j in 1:length(algs)) {\n"
"            truthSample <- readImageData(EXT[EXT[,column] == algs[[j]],])\n"
"            predictedSample <- readImageData(EXP[EXP[,column] == algs[[j]],])\n"
"            for (i in 1:length(predictedSample)) {\n"
"                multiplot(plotImage(predictedSample[[i]], sprintf(\"%s\nPredicted Landmarks\", algs[[j]]), sprintf(\"Average Landmark Error: %.3f\", predictedSample[[i]]$value)), plotImage(truthSample[[i]], \"Ground Truth\nLandmarks\", truthSample[[i]]$path), cols=2)\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"readImageData <- function(data) {\n"
"    examples <- list()\n"
"    for (i in 1:nrow(data)) {\n"
"        examplePath <- unlist(strsplit(data[i,1], \"[:]\"))[1]\n"
"        path <- unlist(strsplit(data[i,1], \"[:]\"))[2]\n"
"        value <- data[i,2]\n"
"        ext <- unlist(strsplit(examplePath, \"[.]\"))[2]\n"
"        if (ext == \"jpg\" || ext == \"JPEG\" || ext == \"jpeg\" || ext == \"JPG\") {\n"
"            img <- readJPEG(examplePath)\n"
"        } else if (ext == \"PNG\" || ext == \"png\") {\n"
"            img <- readPNG(examplePath)\n"
"        } else if (ext == \"TIFF\" || ext == \"tiff\" || ext == \"TIF\" || ext == \"tif\") {\n"
"            img <- readTIFF(examplePath)\n"
"        }else {\n"
"            next\n"
"        }\n"
"        example <- list(path = path, value = value, image = img)\n"
"        examples[[i]] <- example\n"
"    }\n"
"    return(examples)\n"
"}\n"
"\n"
"plotImage <- function(image, title=NULL, label=NULL) {\n"
"    p <- qplot(1:10, 1:10, geom=\"blank\") + annotation_custom(rasterGrob(image$image), xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf)\n"
"    p <- p + theme(axis.line=element_blank(), axis.title.y=element_blank(), axis.text.x=element_blank(), axis.text.y=element_blank(), line=element_blank(), axis.ticks=element_blank(), panel.background=element_blank())\n"
"    p <- p + labs(title=title) + xlab(label)\n"
"    return(p)\n"
"}\n\n");

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
        file.write(qPrintable(plottingUtils));
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
    p.file.write(qPrintable(QString("algs <- %1\n").arg((p.major.size > 1 && p.minor.size > 1) && !(p.major.smooth || p.minor.smooth) ? QString("paste(FF$%1, FF$%2, sep=\"_\")").arg(p.major.header, p.minor.header)
                                                                                                                                      : QString("FF$%1").arg(p.major.size > 1 ? p.major.header : (p.minor.header.isEmpty() ? p.major.header : p.minor.header)))));
    p.file.write("algs <- algs[!duplicated(algs)]\n");

    if (p.major.smooth || p.minor.smooth) {
        QString groupvar = p.major.size > 1 ? p.major.header : (p.minor.header.isEmpty() ? p.major.header : p.minor.header);
        foreach(const QString &data, QStringList() << "DET" << "IET" << "CMC" << "FF" << "FT" << "CT") {
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
        p.file.write("plotFRR(tableData=FF)\n");
        p.file.write("plotFRR(tableData=FF,operatingPoint=1e-6)\n");
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
    p.file.write(qPrintable(QString("algs <- %1\n").arg((p.major.size > 1 && p.minor.size > 1) && !(p.major.smooth || p.minor.smooth) ? QString("paste(FF$%1, FF$%2, sep=\"_\")").arg(p.major.header, p.minor.header)
                                                                                                                                      : QString("FF$%1").arg(p.major.size > 1 ? p.major.header : (p.minor.header.isEmpty() ? p.major.header : p.minor.header)))));
    p.file.write("algs <- algs[!duplicated(algs)]\n");
    if (p.major.smooth || p.minor.smooth) {
        QString groupvar = p.major.size > 1 ? p.major.header : (p.minor.header.isEmpty() ? p.major.header : p.minor.header);
        foreach(const QString &data, QStringList() << "DET" << "FF" << "FT") {
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
