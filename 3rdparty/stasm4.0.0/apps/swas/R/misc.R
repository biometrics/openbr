# misc.R: miscellaneous R definitions

stop0 <- function(...) 
    stop(..., call.=FALSE)

printf <- function(format, ...) # like c printf
	cat(sprintf(format, ...)) 

# Like text() but with a white box underneath to hide stuff under the text.
# For now we ignore font and family when calculating the size of the box.
# TODO this needs work, would be nice to allow left alignment

boxed.text <- function(x, y, labels, cex, pos=1, ...) {
    w <- .5 * max(strwidth(s=labels, cex=cex))
    h <- .5 * max(strheight(s=labels, cex=cex))
    stopifnot(pos == 1 || pos == 3) # 1 is below, 3 is above
    if(pos == 1) # text is below xy
        rect(x - 1.05 * w, y - 1.4 * h,
             x + 1.05 * w, y - 3.8 * h, col="white", border=0)
    else if(pos == 3) # text is above xy
        rect(x - 1.05 * w, y + 1.4 * h,
             x + 1.05 * w, y + 3.8 * h, col="white", border=0)
    text(x=x, y=y, labels=labels, cex=cex, pos=pos, ...)
}
# Lighten color by amount 0 ... 1 where 1 is white.
# If amount is negative, then darken the color, -1 is black.
# Lifted from our internal version of the plotpc package.

lighten <- function(col, lighten.amount, alpha=1)
{
    stopifnot(lighten.amount >= -1 && lighten.amount <= 1)
    rgb <- col2rgb(col) / 255
    # empirically, sqrt makes visual effect of lighten.amount more linear
    lighten.amount2 <- sqrt(abs(lighten.amount))
    rgb <-
        if(lighten.amount > 0)
            rgb + lighten.amount2 * (c(1,1,1) - rgb) # move each r,g,b towards 1
        else # darken
            rgb - lighten.amount2 * rgb              # move each r,g,b towards 0
    if(alpha == 1)
        rgb(rgb[1,], rgb[2,], rgb[3,])
    else
        rgb(rgb[1,], rgb[2,], rgb[3,], alpha)
}
# Strip the common prefix from a vector of strings
# e.g if s is ab12 ab34 ab56 return 12 34 56

strip.common.prefix <- function(s)
{
	all.match <- function(s) {
		if(length(s) > 1)
			for(i in 2:length(s))
				if(s[i] != s[1])
					return(FALSE)
		return(TRUE);
	}
	s <- tolower(s)
	n <- min(nchar(s))
	if(n <= 1)
		return(s)
	for(i in 1:n)
		if(!all.match(substr(s, 1, i)))
			break
	substr(s, i, 1000)
}
get.common.prefix <- function(s)
{
	common <- substr(s[1], 1, max(nchar(s[1]) - nchar(strip.common.prefix(s)[1]), 1))
	n <- nchar(common)
	# strip final _ or space
	if(n > 1 && (substr(common, n, n) == '_' || substr(common, n, n) == ' ')) 
		common <- substr(common, 1, n-1)
	return(common)
}
