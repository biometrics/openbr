# Load libraries
library("ggplot2")
library("gplots")
library("reshape")
library("scales")

# Code to format FAR values
far_names <- list('0.001'="FAR = 0.1%", '0.01'="FAR = 1%")
far_labeller <- function(variable,value) { return(far_names[as.character(value)]) }

plotMetadata <-function(data, title) {
    MT <- as.data.frame(Metadata[c(1, 2, 3, 4, 5),])
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

plotTable <- function(data, name, labels) {
    if (nrow(data) == 0) return()
    if (errBars) {
        input = paste(as.character(round(data$Y, 3)), round(data$ci, 3), sep="\u00b1")
    } else {
        input = data$Y
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

evalFormatting <- function() {
	# Split data into individual plots
    plot_index <<- which(names(data)=="Plot")
    Metadata <<- data[grep("Metadata",data$Plot),-c(1)]
    IM <<- data[grep("IM",data$Plot),-c(1)]
    GM <<- data[grep("GM",data$Plot),-c(1)]
    DET <<- data[grep("DET",data$Plot),-c(1)]
    IET <<- data[grep("IET",data$Plot),-c(1)]
    FAR <- data[grep("FAR",data$Plot),-c(1)]
    FRR <- data[grep("FRR",data$Plot),-c(1)]
    SD <<- data[grep("SD",data$Plot),-c(1)]
    TF <<- data[grep("TF",data$Plot),-c(1)]
    FT <<- data[grep("FT",data$Plot),-c(1)]
    CT <<- data[grep("CT",data$Plot),-c(1)]
    BC <<- data[grep("BC",data$Plot),-c(1)]
    TS <<- data[grep("TS",data$Plot),-c(1)]
    CMC <<- data[grep("CMC",data$Plot),-c(1)]
    FAR$Error <- "FAR"
    FRR$Error <- "FRR"
    ERR <<- rbind(FAR, FRR)
    
    # Format data
    Metadata$Y<-factor(Metadata$Y, levels=c("Genuine", "Impostor", "Ignored", "Gallery", "Probe"))
    IM$Y <<- as.character(IM$Y)
    GM$Y <<- as.character(GM$Y)
    DET$Y <<- as.numeric(as.character(DET$Y))
    IET$Y <<- as.numeric(as.character(IET$Y))
    ERR$Y <<- as.numeric(as.character(ERR$Y))
    SD$Y <<- as.factor(unique(as.character(SD$Y)))
    TF$Y <<- as.numeric(as.character(TF$Y))
    FT$Y <<- as.numeric(as.character(FT$Y))
    CT$Y <<- as.numeric(as.character(CT$Y))
    BC$Y <<- as.numeric(as.character(BC$Y))
    TS$Y <<- as.character(TS$Y)
    CMC$Y <<- as.numeric(as.character(CMC$Y))
}

summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE, conf.interval=0.95, .drop=TRUE) {
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

	datac$upper <- if(datac[, measurevar] + datac$ci < 1) (datac[, measurevar] + datac$ci) else 1
	datac$lower <- if(datac[, measurevar] - datac$ci > 0) (datac[, measurevar] - datac$ci) else 0

	return(datac)
}

plotLine <- function(data=NULL, x=X, y=Y, options=list()) {
    p <- qplot(x, y, data=data, geom="line")
    return(p)
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

