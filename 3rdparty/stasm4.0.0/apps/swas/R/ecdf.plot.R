# ecdf.plot.R: Enhanced version of plot.ecdf, allows different lty's etc.
#              Most of this is copied from Martin Maechler's code.

ecdf.plot <- function(x, xlim, ylim, type,
             xlab = "x", ylab = "proportion",
             # plot horitzonatl lines at quantiles, ygrid=NA to not do that
             ygrid=.5, ygrid.col="gray", ygrid.lty=1,
             # plot vertical lines at quantiles, quantiles=NA to not do that
             quantiles=NA, quantiles.col=ygrid.col, quantiles.lty=ygrid.lty,
             add = FALSE,
             ...)
{
    if(!is.stepfun(x)) {
        if(!is.numeric(x) || !is.vector(x))
            stop("argument x must be a stepfun or a numeric vector")
        x <- ecdf(x)
    }
    knF <- knots(x)
    if (missing(xlim)) {
        rx <- range(knF)
        dr <-
            if(length(knF) > 1L)
                max(0.08 * diff(rx), median(diff(knF)))
            else
                abs(knF)/16
        xlim <- rx +  dr * c(-1,1)
    } else dr <- diff(xlim)
    knF <- knF[xlim[1L]-dr <= knF & knF <= xlim[2L]+dr]
    ti <- c(xlim[1L]-dr, knF, xlim[2L]+dr)
    ti.l <- ti[-length(ti)]
    ti.r <- ti[-1L]
    y <- x(0.5*(ti.l + ti.r))
    # start new code
    x1 <- c(-1e10, knF[1], knF, 1e10)
    y <- c(0, y, 1)
    if (add)
        lines(x1, y, ...)
    else {
        if(missing(ylim)) ylim <- range(c(y,x(knF)))
        if(missing(type)) type <- "l"
        plot(x1, y, type=type, xlim=xlim, ylim=ylim, xlab=xlab, ylab=ylab, ...)
        if(!is.na(ygrid[1])) {
            stopifnot(is.numeric(ygrid))
            stopifnot(all(ygrid >= 0 & ygrid <= 1))
            abline(h=ygrid, col=ygrid.col, lty=ygrid.lty)
        }
        if(!is.na(quantiles[1])) {
            stopifnot(is.numeric(quantiles))
            stopifnot(all(quantiles >= 0 & quantiles <= 1))
            q <- quantile(x, probs=quantiles)
            for(i in seq_along(q))
                segments(q[i], -99,  q[i], quantiles[i],
                         lty=quantiles.lty, col=quantiles.col)
        }
        lines(x1, y, ...) # replot over the grid
    }
    # end new code
    invisible(list(t = ti, y = y))
}
