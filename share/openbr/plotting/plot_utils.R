# Load libraries
library("ggplot2")
library("gplots")
library("reshape")
library("scales")
library("jpeg")
library("png")
library("grid")

# Code to format FAR values
far_names <- list('0.0001'="FAR = 0.01%", '0.001'="FAR = 0.1%", '0.01'="FAR = 1%")
far_labeller <- function(variable,value) {
    if (as.character(value) %in% names(far_names)) {
        return(far_names[as.character(value)])
    } else {
        return(as.character(value))
    }
}

getScale <- function(mode, title, vals) {
    if      (vals > 12) return(do.call(paste("scale", mode, "discrete", sep="_"), list(title)))
    else if (vals > 11) return(do.call(paste("scale", mode, "brewer", sep="_"), list(title, palette=(if(seq) "Reds" else "Set3"), type=(if(seq) "seq" else "qual"))))
    else if (vals > 9)  return(do.call(paste("scale", mode, "brewer", sep="_"), list(title, palette=(if(seq) "Reds" else "Paired"), type=(if(seq) "seq" else "qual"))))
    else                return(do.call(paste("scale", mode, "brewer", sep="_"), list(title, palette=(if(seq) "Reds" else "Set1"), type=(if(seq) "seq" else "qual"))))
}

plotMetadata <- function(metadata=NULL, title=NULL) {
    MT <- as.data.frame(metadata[c(1, 2, 3, 4, 5),])
    par(mfrow=c(4, 1))
    plot.new()
    print(title(paste(title, date(), sep="\n")))
    mat <- matrix(MT$X[c(1, 2)], ncol=2)
    colnames(mat) <- c("Gallery", "Probe")
    imageTable <- as.table(mat)
    print(textplot(imageTable, show.rownames=FALSE))
    print(title("Images"))
    mat <- matrix(MT$X[c(3, 4, 5)], ncol=3)
    colnames(mat) <- c("Genuine", "Impostor", "Ignored")
    matchTable <- as.table(mat)
    print(textplot(matchTable, show.rownames=FALSE))
    print(title("Matches"))
    plot.new()
    print(title("Gallery * Probe = Genuine + Impostor + Ignored"))
}

plotTable <- function(tableData=NULL, name=NULL, labels=NULL) {
    if (nrow(tableData) == 0) return()
    if (smooth && confidence != 0) {
        input = paste(as.character(round(tableData$Y, 3)), round(tableData$ci, 3), sep="\u00b1")
    } else {
        input = tableData$Y
    }
    mat <- matrix(input, nrow=length(labels), ncol=length(algs), byrow=FALSE)
    colnames(mat) <- algs
    rownames(mat) <- labels
    table <- as.table(mat)
    if (csv) {
        write.csv(table, file=paste(paste(basename, deparse(substitute(data)), sep="_"), ".csv", sep=""))
    } else {
        print(textplot(table))
        print(title(name))
    }
}

plotTAR <- function(tableData=NULL, operatingPoint=1e-4) {
    if (nrow(tableData) == 0) return()
    major <- majorHeader
    minor <- if(minorHeader == "") majorHeader else minorHeader
    tableData <- tableData[grep(operatingPoint,tableData$X),]
    mat <- matrix(tableData$Y, nrow=length(tableData[,major][!duplicated(tableData[,major])]), ncol=length(tableData[,minor][!duplicated(tableData[,minor])]), byrow=FALSE)
    colnames(mat) <- tableData[,minor][!duplicated(tableData[,minor])]
    rownames(mat) <- tableData[,major][!duplicated(tableData[,major])]
    table <- as.table(mat)
    print(textplot(table))
    print(title(paste("True Accept Rate at FAR=", toString(operatingPoint))))
}

plotLandmarkTables <- function(tableData=NULL) {
    if (majorSize > 1) {
        var <- majorHeader
    } else {
        if (minorHeader == "") var <- majorHeader else var <- minorHeader
    }
    StatBox <- summarySE(tableData, measurevar="Y", groupvars=c(var,"X"))
    OverallStatBox <- summarySE(tableData, measurevar="Y", groupvars=c(var))
    mat <- matrix(paste(as.character(round(StatBox$Y, 3)), round(StatBox$ci, 3), sep=" \u00b1 "), nrow=rows, ncol=length(algs), byrow=FALSE)
    mat <- rbind(mat, paste(as.character(round(OverallStatBox$Y, 3)), round(OverallStatBox$ci, 3), sep=" \u00b1 "))
    mat <- rbind(mat, as.character(round(NormLength$Y, 3)))
    colnames(mat) <- algs
    rownames(mat) <- c(seq(0, rows-1), "Aggregate","Average IPD")
    ETable <- as.table(mat)
    print(textplot(ETable))
    print(title("Landmarking Error Rates"))
}

plotLine <- function(lineData=NULL, options=NULL, flipY=FALSE, geometry="path") {
    textSize <- if("textSize" %in% names(options)) as.numeric(options$textSize) else 12
    p <- qplot(X, if(flipY) 1-Y else Y, data=lineData, main=options$title, geom=geometry, size=if("size" %in% names(options)) I(as.numeric(options$size)) else I(.5), colour=if(majorSize > 1) factor(eval(parse(text=majorHeader))) else NULL, linetype=if(minorSize > 1) factor(eval(parse(text=minorHeader))) else NULL, xlab=options$xTitle, ylab=options$yTitle) + theme_minimal()
    if (smooth && deparse(substitute(lineData)) != "CMC" && confidence != 0) p <- p + geom_errorbar(data=lineData[seq(1, NROW(lineData), by = 29),], aes(x=X, ymin=if(flipY) (1-lower) else lower, ymax=if(flipY) (1-upper) else upper), width=0.1, alpha=I(1/2))
    if (majorSize > 1) p <- p + getScale("colour", majorHeader, majorSize)
    if (minorSize > 1) p <- p + scale_linetype_discrete(minorHeader)

    # Set log/continuous scales, breaks and labels
    if (options$xLog)
        p <- p + scale_x_log10(labels=if("xLabels" %in% names(options)) eval(parse(text=options$xLabels)) else trans_format("log10", math_format()), breaks=if("xBreaks" %in% names(options)) eval(parse(text=options$xBreaks)) else waiver()) + annotation_logticks(sides="b")
    else
        p <- p + scale_x_continuous(labels=if("xLabels" %in% names(options)) eval(parse(text=options$xLabels)) else percent, breaks=if("xBreaks" %in% names(options)) eval(parse(text=options$xBreaks)) else pretty_breaks(n=10))
    if (options$yLog)
        p <- p + scale_y_log10(labels=if("yLabels" %in% names(options)) eval(parse(text=options$yLabels)) else trans_format("log10", math_format()), breaks=if("yBreaks" %in% names(options)) eval(parse(text=options$yBreaks)) else waiver()) + annotation_logticks(sides="l")
    else
        p <- p + scale_y_continuous(labels=if("yLabels" %in% names(options)) eval(parse(text=options$yLabels)) else percent, breaks=if("yBreaks" %in% names(options)) eval(parse(text=options$yBreaks)) else pretty_breaks(n=10))

    if ("xLimits" %in% names(options) && "yLimits" %in% names(options)) {
        p <- p + coord_cartesian(xlim=eval(parse(text=options$xLimits)), ylim=eval(parse(text=options$yLimits)))
    } else {
        if ("xLimits" %in% names(options)) p <- p + coord_cartesian(xlim=eval(parse(text=options$xLimits)))
        if ("yLimits" %in% names(options)) p <- p + coord_cartesian(ylim=eval(parse(text=options$yLimits)))
    }
    p <- p + theme(legend.title = element_text(size = textSize), legend.text = element_text(size = textSize), plot.title = element_text(size = textSize), axis.text = element_text(size = textSize), axis.title.x = element_text(size = textSize), axis.title.y = element_text(size = textSize), legend.position=if("legendPosition" %in% names(options)) eval(parse(text=options$legendPosition)) else "bottom", legend.background = element_rect(fill = 'white'), panel.grid.major = element_line(colour = "gray"), panel.grid.minor = element_line(colour = "gray", linetype = "dashed"))
    p <- p + guides(colour=guide_legend(ncol=ncol)) + guides(linetype=guide_legend(ncol=ncol))
    return(p)
}

plotSD <- function(sdData=NULL) {
    p <- qplot(X, data=sdData, geom="histogram", fill=Y, position="identity", alpha=I(1/2), xlab="Score", ylab="Frequency")
    p <- p + scale_fill_manual("Ground Truth", values=c("blue", "red")) + theme_minimal() + scale_x_continuous(minor_breaks=NULL) + scale_y_continuous(minor_breaks=NULL) + theme(axis.text.y=element_blank(), axis.ticks=element_blank(), axis.text.x=element_text(angle=-90, hjust=0))
    if (majorSize > 1) {
        if (minorSize > 1) {
            if (flip) {
                A <- minorHeader
                B <- majorHeader
            } else {
                A <- majorHeader
                B <- minorHeader
            }
            p <- p + facet_grid(facets=as.formula(paste(A, "~", B)), scales="free")
        } else {
            p <- p + facet_wrap(facets=as.formula(paste("~", majorHeader)), scales="free")
        }
    }
    p <- p + theme(aspect.ratio=1)
    return(p)
}

plotBC <- function(bcData=NULL) {
    factor <- if (majorSmooth) minorHeader else majorHeader
    plotString <- paste("qplot(factor(", factor, ")", if(smooth) ", Y" else "", ", data=bcData, ", if(smooth) "geom=\"boxplot\"" else "geom=\"bar\", position=\"dodge\", weight=Y", sep="")
    p <- eval(parse(text=paste(plotString, if(majorSize > 1) paste(", fill=factor(", majorHeader, ")", sep="") else "", ", xlab=\"False Accept Rate\", ylab=\"True Accept Rate\") + theme_minimal()", sep="")))
    if (majorSize > 1) p <- p + getScale("fill", majorHeader, majorSize)
    if (minorSize > 1) p <- p + facet_grid(facets=as.formula(paste(minorHeader, "~", "X")), labeller=far_labeller) else p <- p + facet_grid(. ~ X, labeller=far_labeller)
    p <- p + scale_y_continuous(labels=percent) + theme(legend.position="none", axis.text.x=element_text(angle=-90, hjust=0))
    if (!smooth) p <- p + geom_text(data=bcData, aes(label=Y, y=0.05))
    return(p)
}

plotOverlap <- function(overlapData=NULL) {
    p <- qplot(X, data=overlapData, geom="histogram", position="identity", xlab="Overlap", ylab="Frequency")
    p <- p + theme_minimal() + scale_x_continuous(minor_breaks=NULL) + scale_y_continuous(minor_breaks=NULL) + theme(axis.text.y=element_blank(), axis.ticks=element_blank(), axis.text.x=element_text(angle=-90, hjust=0))
    if (majorSize > 1) {
        if (minorSize > 1) {
            p <- p + facet_grid(facets=as.formula(paste(minorHeader, "~", majorHeader)), scales="free")
        } else {
            p <- p + facet_wrap(facets=as.formula(paste("~", majorHeader)), scales="free")
        }
    }
    p <- p + theme(aspect.ratio=1, legend.position="bottom")
    return(p)
}

formatData <- function(type="eval") {
    if (type == "eval") {
	    # Split data into individual plots
        plot_index <<- which(names(data)=="Plot")
        Metadata <<- data[grep("Metadata",data$Plot),-c(1)]
        IM <<- data[grep("IM",data$Plot),-c(1)]
        GM <<- data[grep("GM",data$Plot),-c(1)]
        DET <<- data[grep("DET",data$Plot),-c(1)]
        IET <<- data[grep("IET",data$Plot),-c(1)]
        FAR <<- data[grep("FAR",data$Plot),-c(1)]
        FRR <<- data[grep("FRR",data$Plot),-c(1)]
        SD <<- data[grep("SD",data$Plot),-c(1)]
        TF <<- data[grep("TF",data$Plot),-c(1)]
        FT <<- data[grep("FT",data$Plot),-c(1)]
        CT <<- data[grep("CT",data$Plot),-c(1)]
        BC <<- data[grep("BC",data$Plot),-c(1)]
        TS <<- data[grep("TS",data$Plot),-c(1)]
        CMC <<- data[grep("CMC",data$Plot),-c(1)]
    
        # Format data
        Metadata$Y<-factor(Metadata$Y, levels=c("Genuine", "Impostor", "Ignored", "Gallery", "Probe"))
        IM$Y <<- as.character(IM$Y)
        GM$Y <<- as.character(GM$Y)
        DET$Y <<- as.numeric(as.character(DET$Y))
        IET$Y <<- as.numeric(as.character(IET$Y))
        FAR$Y <<- as.numeric(as.character(FAR$Y))
        FRR$Y <<- as.numeric(as.character(FRR$Y))
        SD$Y <<- as.factor(unique(as.character(SD$Y)))
        TF$Y <<- as.numeric(as.character(TF$Y))
        FT$Y <<- as.numeric(as.character(FT$Y))
        CT$Y <<- as.numeric(as.character(CT$Y))
        BC$Y <<- as.numeric(as.character(BC$Y))
        TS$Y <<- as.character(TS$Y)
        CMC$Y <<- as.numeric(as.character(CMC$Y))
    } else if (type == "detection") {
        # Split data into individual plots
        DiscreteROC <<- data[grep("DiscreteROC",data$Plot),-c(1)]
        ContinuousROC <<- data[grep("ContinuousROC",data$Plot),-c(1)]
        DiscretePR <<- data[grep("DiscretePR",data$Plot),-c(1)]
        ContinuousPR <<- data[grep("ContinuousPR",data$Plot),-c(1)]
        Overlap <<- data[grep("Overlap",data$Plot),-c(1)]
        AverageOverlap <<- data[grep("AverageOverlap",data$Plot),-c(1)]
    } else if (type == "landmarking") {
        # Split data into individual plots
        Box <<- data[grep("Box",data$Plot),-c(1)]
        Box$X <<- factor(Box$X, levels = Box$X, ordered = TRUE)
        Sample <<- data[grep("Sample",data$Plot),-c(1)]
        Sample$X <<- as.character(Sample$X)
        displaySample <<- readImageData(Sample)
        rows <<- displaySample[[1]]$value
        EXT <<- data[grep("EXT",data$Plot),-c(1)]
        EXT$X <<- as.character(EXT$X)
        EXP <<- data[grep("EXP",data$Plot),-c(1)]
        EXP$X <<- as.character(EXP$X)
        NormLength <<- data[grep("NormLength",data$Plot),-c(1)]
    } else if (type == "knn") {
        # Split data into individual plots
        IET <<- data[grep("IET",data$Plot),-c(1)]
        IET$Y <<- as.numeric(as.character(IET$Y))
        CMC <<- data[grep("CMC",data$Plot),-c(1)]
        CMC$Y <<- as.numeric(as.character(CMC$Y))
    }
}

summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE, conf.interval=0.95, .drop=TRUE) {
    # derived from http://www.cookbook-r.com/Manipulating_data/Summarizing_data/
	require(plyr)

	length2 <- function (x, na.rm=FALSE) {
		if (na.rm) sum(!is.na(x))
		else       length(x)
	}

	datac <- ddply(data, groupvars, .drop=.drop, .fun = function(xx, col) {
		c(N=length2(xx[[col]], na.rm=na.rm), mean=mean(xx[[col]], na.rm=na.rm), sd=sd(xx[[col]], na.rm=na.rm))
		},
		measurevar
	)

	datac <- rename(datac, c("mean" = measurevar))
	datac$se <- datac$sd / sqrt(datac$N)
	ciMult <- qt(conf.interval/2 + .5, datac$N-1)
	datac$ci <- datac$se * ciMult

	datac$upper <- ifelse(datac[,measurevar] + datac$ci < 1, datac[,measurevar] + datac$ci, 1)
	datac$lower <- ifelse(datac[,measurevar] - datac$ci > 0, datac[,measurevar] - datac$ci, 0)

	return(datac)
}

multiplot <- function(..., plotlist=NULL, cols) {
    require(grid)
    # Make a list from the ... arguments and plotlist
    plots <- c(list(...), plotlist)
    numPlots = length(plots)
    # Make the panel
    plotCols = cols
    plotRows = ceiling(numPlots/plotCols)
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(plotRows, plotCols)))
    vplayout <- function(x, y)
    viewport(layout.pos.row = x, layout.pos.col = y)
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
        curRow = ceiling(i/plotCols)
        curCol = (i-1) %% plotCols + 1
        print(plots[[i]], vp = vplayout(curRow, curCol))
    }
}

plotEERSamples <- function(imData=NULL, gmData=NULL) {
    if(nrow(imData) == 0) return()

    printImages <- function(images, label) {
        for (i in 1:nrow(images)) {
            score <- images[i,1]
            files <- images[i,2]
            alg <- images[i,3]
            files <- unlist(strsplit(files, "[:]"))

            ext1 <- unlist(strsplit(files[2], "[.]"))[2]
            ext2 <- unlist(strsplit(files[4], "[.]"))[2]
            if (ext1 == "jpg" || ext1 == "JPEG" || ext1 == "jpeg" || ext1 == "JPG") {
                img1 <- readJPEG(files[2])
            } else if (ext1 == "PNG" || ext1 == "png") {
                img1 <- readPNG(files[2])
            } else if (ext1 == "TIFF" || ext1 == "tiff" || ext1 == "TIF" || ext1 == "tif") {
                img1 <- readTIFF(files[2])
            } else {
                next
            }
            if (ext2 == "jpg" || ext2 == "JPEG" || ext2 == "jpeg" || ext2 == "JPG") {
                img2 <- readJPEG(files[4])
            } else if (ext2 == "PNG" || ext2 == "png") {
                img2 <- readPNG(files[4])
            } else if (ext2 == "TIFF" || ext2 == "tiff" || ext2 == "TIF" || ext2 == "tif") {
                img2 <- readTIFF(files[4])
            } else {
                next
            }
            name1 <- files[1]
            name2 <- files[3]

            g1 <- rasterGrob(img1, interpolate=TRUE)
            g2 <- rasterGrob(img2, interpolate=TRUE)

            plot1 <- qplot(1:10, 1:10, geom="blank") + annotation_custom(g1, xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) + theme(axis.line=element_blank(), axis.text.x=element_blank(), axis.text.y=element_blank(), axis.ticks=element_blank(), panel.background=element_blank()) + labs(title=alg) + ylab(unlist(strsplit(files[2], "[/]"))[length(unlist(strsplit(files[2], "[/]")))]) + xlab(name1)
            plot2 <- qplot(1:10, 1:10, geom="blank") + annotation_custom(g2, xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) + theme(axis.line=element_blank(), axis.text.x=element_blank(), axis.text.y=element_blank(), axis.ticks=element_blank(), panel.background=element_blank()) + labs(title=paste(label, " score =", score)) + ylab(unlist(strsplit(files[4], "[/]"))[length(unlist(strsplit(files[4], "[/]")))]) + xlab(name2)

            multiplot(plot1, plot2, cols=2)
        }
    }
    printImages(imData, "Impostor")
    printImages(gmData, "Genuine")
}

plotLandmarkSamples <- function(displaySample=NULL, expData=NULL, extData=NULL) {
    print(plotImage(displaySample[[1]], "Sample Landmarks", sprintf("Total Landmarks: %s", displaySample[[1]]$value)))
    column <- if(majorSize > 1) majorHeader else minorHeader
    if (nrow(EXT) != 0 && nrow(EXP)) {
        for (j in 1:length(algs)) {
            truthSample <- readImageData(EXT[EXT[,column] == algs[[j]],])
            predictedSample <- readImageData(EXP[EXP[,column] == algs[[j]],])
            for (i in 1:length(predictedSample)) {
                multiplot(plotImage(predictedSample[[i]], sprintf("%s\nPredicted Landmarks", algs[[j]]), sprintf("Average Landmark Error: %.3f", predictedSample[[i]]$value)), plotImage(truthSample[[i]], "Ground Truth\nLandmarks", truthSample[[i]]$path), cols=2)
            }
        }
    }
}

readImageData <- function(data) {
    examples <- list()
    for (i in 1:nrow(data)) {
        examplePath <- unlist(strsplit(data[i,1], "[:]"))[1]
        path <- unlist(strsplit(data[i,1], "[:]"))[2]
        value <- data[i,2]
        ext <- unlist(strsplit(examplePath, "[.]"))[2]
        if (ext == "jpg" || ext == "JPEG" || ext == "jpeg" || ext == "JPG") {
            img <- readJPEG(examplePath)
        } else if (ext == "PNG" || ext == "png") {
            img <- readPNG(examplePath)
        } else if (ext == "TIFF" || ext == "tiff" || ext == "TIF" || ext == "tif") {
            img <- readTIFF(examplePath)
        }else {
            next
        }
        example <- list(path = path, value = value, image = img)
        examples[[i]] <- example
    }
    return(examples)
}

plotImage <- function(image, title=NULL, label=NULL) {
    p <- qplot(1:10, 1:10, geom="blank") + annotation_custom(rasterGrob(image$image), xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf)
    p <- p + theme(axis.line=element_blank(), axis.title.y=element_blank(), axis.text.x=element_blank(), axis.text.y=element_blank(), line=element_blank(), axis.ticks=element_blank(), panel.background=element_blank())
    p <- p + labs(title=title) + xlab(label)
    return(p)
}

