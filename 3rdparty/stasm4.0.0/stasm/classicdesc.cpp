// classicdesc.cpp: model for classic ASM descriptors
//
//     By "classic descriptor" we mean the Cootes' style one dimensional
//     profile along the whisker orthogonal to the shape boundary.
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"

namespace stasm
{
static void NormalizeMat(
    MAT&   mat)            // io: normalized so L2 length is 1
{
    double norm = cv::norm(mat); // L2 norm
    if (!IsZero(norm))
        mat /= norm;
}

static const VEC Bisector( // return normalized bisector of three ordered points
    const VEC& prev,       // in: x,y coords of previous point (1x2 matrix)
    const VEC& point,      // in
    const VEC& next)       // in
{
    VEC u(1, 2);              // u is point - prev, rotated by 90 degrees
    u(IX) = point(IY) - prev(IY);
    u(IY) = prev(IX) - point(IX);
    NormalizeMat(u);

    VEC v(1, 2);              // v is next - point, rotated by 90 degrees
    v(IX) = next(IY) - point(IY);
    v(IY) = point(IX) - next(IX);
    NormalizeMat(v);

    VEC w(u + v); NormalizeMat(w);

    // are prev and next in the same line? if so, avoid numerical issues
    if (IsZero(w(IX)) && IsZero(w(IY)))
    {
        w = point - prev;
        NormalizeMat(w);
    }
    return w; // w is the direction of the bisector
}

// get x and y distances to take a single pixel step along the whisker

static void WhiskerStep(
    double&      xstep,  // out: x dist to move one pix along whisker
    double&      ystep,  // out: y dist to move one pix along whisker
    const Shape& shape,  // in
    int          ipoint) // in: index of the current point
{
    int prev, next; PrevAndNextLandmarks(prev, next, ipoint, shape);

    if ((Equal(shape(prev, IX), shape(ipoint, IX)) &&
         Equal(shape(prev, IY), shape(ipoint, IY)))     ||

        (Equal(shape(next, IX), shape(ipoint, IX)) &&
         Equal(shape(next, IY), shape(ipoint, IY))))
    {
        // The prev or next point is on top of the current point.
        // Arbitrarily point the whisker in a horizontal direction.
        // TODO Revisit, this is common at low resolution pyramid levels.

        xstep = 1;
        ystep = 0;
    }
    else
    {
        const VEC whisker_direction(Bisector(shape.row(prev),
                                             shape.row(ipoint),
                                             shape.row(next)));
        xstep = -whisker_direction(IX);
        ystep = -whisker_direction(IY);

        // normalize so either xstep or ystep will be +-1,
        // and the other will be smaller than +-1

        const double abs_xstep = ABS(xstep);
        const double abs_ystep = ABS(ystep);
        if (abs_xstep >= abs_ystep)
        {
            xstep /= abs_xstep;
            ystep /= abs_xstep;
        }
        else
        {
            xstep /= abs_ystep;
            ystep /= abs_ystep;
        }
    }
}

static inline int Step( // return x coord at the given offset along whisker
    double x,           // in: x coord of center of whisker
    double xstep,       // in: x dist to move one pixel along whisker
    int    offset)      // in: offset along whisker in pixels
{
    return cvRound(x + (offset * xstep));
}

static inline int Pix( // get pixel at ix and iy, forcing ix and iy in range
    const Image& img,  // in
    int          ix,   // in
    int          iy)   // in
{
    return img(Clamp(iy, 0, img.rows-1), Clamp(ix, 0, img.cols-1));
}

// fullprof is the 1D profile along the whisker, including extra elements
// to allow searching away from the current position of the landmark.
//
// shape[ipoint] is the current position of the landmark,
// and the center point of the whisker.
// We also use shape for figuring out the direction of the whisker.

static void FullProf(
    VEC&         fullprof, // out
    const Image& img,      // in
    const MAT&   shape,    // in
    int          ipoint)   // in: index of the current point
{
    double xstep; // x axis dist corresponding to one pixel along whisker
    double ystep;
    WhiskerStep(xstep, ystep, shape, ipoint);

    const double x = shape(ipoint, IX); // center point of the whisker
    const double y = shape(ipoint, IY);

    // number of pixs to sample in each direction along the whisker
    const int n = (NSIZE(fullprof) - 1) / 2;

    int prevpix = Pix(img,
                      Step(x, xstep, -n-1), Step(y, ystep, -n-1));

    for (int i = -n; i <= n; i++)
    {
        const int pix = Pix(img,
                            Step(x, xstep, i), Step(y, ystep, i));

        fullprof(i + n) = double(pix - prevpix); // signed gradient
        prevpix = pix;
    }
}

double SumAbsElems( // return the sum of the abs values of the elems of mat
    const MAT& mat) // in
{
    CV_Assert(mat.isContinuous());
    const double* const data = Buf(mat);
    double sum = 0;
    int i = NSIZE(mat); // number of elements
    while (i--)
        sum += ABS(data[i]);
    return sum;
}

// This returns a double equal to x.t() * mat * x.
//
// x is a vector (row or column, it doesn't matter).
//
// mat is assumed to be a symmetric matrix
// (but only the upper right triangle of mat is actually used).
//
// mat and x are not modified.
//
// This function is equivalent to x.t() * mat * x(), but is optimized
// for speed and is faster.  It's faster because we use the fact that
// mat is symmetric to roughly halve the number of operations.

static double xAx(
    const VEC& x,   // in
    const MAT& mat) //in: must be symmetric
{
    const int n = NSIZE(x);
    CV_Assert(mat.rows == n && mat.cols == n && x.isContinuous());
    const double* px = Buf(x);
    double diagsum = 0, sum = 0;
    int i = n;
    while (i--)
    {
        const double xi = px[i];
        const double* const rowbuf = mat.ptr<double>(i);
        diagsum += rowbuf[i] * SQ(xi);    // sum diag elements
        for (int j = i+1; j < n; j++)     // sum upper right triangle elements
            sum += rowbuf[j] * xi * px[j];
    }
    return diagsum + 2 * sum; // "2 *" to include lower left triangle elements
}

// Get the profile distance.  That is,  get the image profile at the given
// offset  along the whisker, and return the Mahalanobis distance between
// it and the model mean profile.  Low distance means good fit.

static double ProfDist(
    int        offset,    // in: offset along whisker in pixels
    int        proflen,   // in
    const VEC& fullprof,  // in
    const VEC& meanprof,  // in: mean of the training profiles for this point
    const MAT& covi)      // in: inverse of the covar of the training profiles
{
    VEC prof(1, proflen); // the profile at the given offset along whisker

    // copy the relevant part of fullprof into prof

    memcpy(Buf(prof),
           Buf(fullprof) + offset + NSIZE(fullprof)/2 - NSIZE(prof)/2,
           NSIZE(prof) * sizeof(prof(0)));

    // normalize prof

    double sum = SumAbsElems(prof);
    if (!IsZero(sum))
        prof *= NSIZE(prof) / sum;

    // The following code is equivalent to
    //      return (prof - meanprof).t() * covi * (prof - meanprof)
    // but is optimized for speed.

    prof -= meanprof;      // for efficiency, use "-=" not "=" with "-"
    return xAx(prof, covi);
}

// If OpenMP is enabled, multiple instances of this function will be called
// concurrently (each call will have a different value of x and y). Thus this
// function and its callees do not modify any data that is not on the stack.

void ClassicDescSearch(    // search along whisker for best profile match
    double&      x,        // io: (in: old posn of landmark, out: new posn)
    double&      y,        // io:
    const Image& img,      // in: the image scaled to this pyramid level
    const Shape& inshape,  // in: current posn of landmarks (for whisker directions)
    int          ipoint,   // in: index of the current landmark
    const MAT&   meanprof, // in: mean of the training profiles for this point
    const MAT&   covi)     // in: inverse of the covar of the training profiles
{
    const int proflen = NSIZE(meanprof);
    CV_Assert(proflen % 2 == 1); // proflen must be odd in this implementation

    // fullprof is the 1D profile along the whisker including the extra
    // elements to allow search +-CLASSIC_MAX_OFFSET pixels away from
    // the current position of the landmark.
    // We precalculate the fullprof for efficiency in the for loop below.

    VEC fullprof(1, proflen + 2 * CLASSIC_MAX_OFFSET);
    CV_Assert(NSIZE(fullprof) % 2 == 1); // fullprof length must be odd
    FullProf(fullprof, img, inshape, ipoint);

    // move along the whisker looking for the best match

    int bestoffset = 0;
    double mindist = FLT_MAX;
    for (int offset = -CLASSIC_MAX_OFFSET;
             offset <= CLASSIC_MAX_OFFSET;
             offset += CLASSIC_SEARCH_RESOL)
    {
        const double dist = ProfDist(offset, proflen, fullprof, meanprof, covi);

        if (dist < mindist)
        {
            mindist = dist;
            bestoffset = offset;
        }
    }
    // change x,y to the best position along the whisker

    double xstep, ystep;
    WhiskerStep(xstep, ystep, inshape, ipoint);
    x = inshape(ipoint, IX) + (bestoffset * xstep);
    y = inshape(ipoint, IY) + (bestoffset * ystep);
}

} // namespace stasm
