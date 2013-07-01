# swas.R: R language program for graphing results of running swas.exe
#
# To use: put the names of the fit file(s) produced by running swas.exe
# into FILENAMES.  Then run this program to get swas.pdf.

library(stats)
library(graphics)
source("../apps/swas/R/misc.R")
source("../apps/swas/R/ecdf.plot.R")

FILENAMES <- c(

"fit/bioid.fit", # create with swas -i ../data/bioid.shape

"" # placeholder so can have a comma terminating the final filename above
)
read.fit.file <- function(filename)
{
    dat <- read.table(filename, header=TRUE, stringsAsFactors=FALSE)
    if(!identical(colnames(dat),
        c("file", "meanfit", "iworst",
          "me17", "fm29", "iworstfm29", "intereye", "eyemouth", 
          "yaw", "pitch", "roll", "poseerr", "dettime", "asmtime")))
        stop0("unexpected colnames in ", filename)
	dat
}
strip.suffix <- function(filename)
{
    gsub("\\.fit", "", filename) # strip .fit from filename
}
get.tags <- function(filenames)
{
	strip.suffix(filenames)
}
open.pdf <- function(pdf.name)
{
    printf("Creating %s\n", pdf.name)
    pdf(pdf.name)
    par(cex=1.2)
    par(mar=c(4, 4, 2, 1)) # small margins and text
    par(mgp=c(2, 0.6, 0))  # flatten axis elements
}
print.summary <- function(ifile, filename, stripped.filesnames, max.stripped.filename)
{
    dat <- read.fit.file(filename)
    q <- quantile(dat$fm29, probs=c(0, .5, .90, .95, 1), names=FALSE, na.rm=TRUE)
	meanfm29 <- mean(dat$fm29, na.rm=TRUE)
	# for consistency, make mean() give same result as median() with all NAs
	if(is.nan(meanfm29)) 
		meanfm29 <- NA
    filename.format <- sprintf("%%-%d.%ds ", max.stripped.filename, max.stripped.filename)
    printf(filename.format, strip.suffix(stripped.filesnames))
	dat$dettime <- gsub("\\[|\\]", "", dat$dettime) # convert [1.23] to 1.23
	dat$asmtime <- gsub("\\[|\\]", "", dat$asmtime)
    printf("%7.0d    ", nrow(dat))
    printf("%5.3f %5.3f %7.3f %7.3f ", meanfm29, q[2], q[5], q[3])
    printf("    %5.3f %5.3f ", median(dat$me17, na.rm=TRUE), mean(dat$me17, na.rm=TRUE))
    printf("%7.3f %7.3f\n", mean(as.numeric(dat$dettime)), mean(as.numeric(dat$asmtime)))
}
get.col <- function(ifile, filename)
{
	PALETTE <- rep(c("black", "gray60", "sienna", "goldenrod2", 
			     "#FF6050", "#FFB090",  "green2", 
                 "#008050", "blue", "magenta"),
               each=1, length.out=1000)

	PALETTE[ifile]
}
plot.me17 <- function(ifile, filename)
{
    dat <- read.fit.file(filename)
	col <- get.col(ifile, filename)
    if (FIRST.FILE) { # if first file, set up graph
		FIRST.FILE <<- FALSE
		xlim <- c(0, .15)
        ecdf.plot(ecdf(dat$me17), col=col,
			xlab="me17", ylab="proportion", 
            xlim=c(0,.15), xaxs="i", yaxs="i",
			ygrid=c(.5,.90), ygrid.col="lightgray",
			main="me17 distribution", cex.main=1)
		grid()
    } else
        ecdf.plot(ecdf(dat$me17), add=TRUE, col=col)
}
plot.legend <- function(filenames)
{
    col <- get.col(1:length(filenames), filenames)
    legend("bottomright", legend=get.tags(FILENAMES),
		    bg="white", col=col, text.col=col, lty=1)
	abline(h=0) # hack: fix bottom line of ecdf graph
}
#--- code starts here ---

# if placeholder exists, drop it
if (identical(FILENAMES[length(FILENAMES)], ""))
    FILENAMES <- FILENAMES[1:(length(FILENAMES)-1)]

open.pdf("swas.pdf")

# print summaries
max.stripped.filename <- max(5, nchar(get.tags(FILENAMES)))
filename.format <- sprintf("%%-%d.%ds ", max.stripped.filename, max.stripped.filename)
printf(filename.format, "file")
printf("nshapes fm29:mean  med     max quant90 me17:mean   med dettime asmtime\n")
FIRST.FILE <- TRUE
for (ifile in 1:length(FILENAMES))
    print.summary(ifile, FILENAMES[ifile], get.tags(FILENAMES)[ifile], max.stripped.filename)

# plot me17 densities
FIRST.FILE <- TRUE
xlim <- c(0, .15)
for (ifile in 1:length(FILENAMES))
    plot.me17(ifile, FILENAMES[ifile])
if(!FIRST.FILE) # plotted at least one file?
	plot.legend(FILENAMES)

dev.off() # close pdf file
