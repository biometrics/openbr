// hat.cpp: Histogram Array Transform descriptors
//
// Rob Hess' opensift implementation was used a reference:
// http://blogs.oregonstate.edu/hess/code/sift
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"

namespace stasm
{
static const int GRIDHEIGHT = 4;       // 4 x 5 grid of histograms in descriptor
static const int GRIDWIDTH  = 5;

static const int BINS_PER_HIST = 8;    // 8 gives a 45 degree range per bin

static const double WINDOW_SIGMA = .5; // gaussian window as frac of patch width
                                       // .5 implies patch border downweight is .368

static const double FINAL_SCALE = 10;  // arb but 10 is good for %g printing of descriptors

// Get gradient magnitude and orientation of pixels in given img.
// We use a [1,-1] convolution mask rather than [1,0,-1] because it gives as good
// Stasm results and doesn't "waste" pixels on the left and top image boundary.
// Before scaling by bins_per_degree, orientations are from 0 to 359.99...
// degrees, with 0 being due east, and anticlockwise increasing.

static void InitGradMagAndOrientMats(
    MAT&         magmat,    // out: grad mag mat
    MAT&         orientmat, // out: grad ori mat
    const Image& img)       // in:  ROI scaled to current pyramid level
{
    const int nrows = img.rows, nrows1 = img.rows-1;
    const int ncols = img.cols, ncols1 = img.cols-1;
    const double bins_per_degree = BINS_PER_HIST / 360.;

    magmat.create(nrows, ncols);
    orientmat.create(nrows, ncols);

    for (int y = 0; y < nrows1; y++)
    {
        const byte* const buf    = (byte*)(img.data) + y     * ncols;
        const byte* const buf_x1 = (byte*)(img.data) + y     * ncols + 1;
        const byte* const buf_y1 = (byte*)(img.data) + (y+1) * ncols;

        double* const magbuf    = Buf(magmat)    + y * ncols;
        double* const orientbuf = Buf(orientmat) + y * ncols;

        for (int x = 0; x < ncols1; x++)
        {
            const byte   pixel  = buf[x];
            const double xdelta = buf_x1[x] - pixel;
            const double ydelta = buf_y1[x] - pixel;

            magbuf[x] = sqrt(SQ(xdelta) + SQ(ydelta));

            double orient =
                RadsToDegrees(atan2(ydelta, xdelta)); // -180 <= orient < 180
            if (orient < 0)
                orient += 360;                        // 0 <= orient < 360
            orientbuf[x] = orient * bins_per_degree;  // 0 <= orient < BINS_PER_HIST
        }
    }
    // fill bottom and right edges
    magmat.row(nrows1) = 0;
    magmat.col(ncols1) = 0;
    orientmat.row(nrows1) = 0;
    orientmat.col(ncols1) = 0;
}

// Init the indices which map a patch row,col to the corresponding
// histogram grid row,col.  The mapping depends only on the image
// patchwidth and the histogram GRIDHEIGHT and WIDTH.
//
// The first pixel in the image patch maps to histogram grid x coord -0.5.
// Therefore after TrilinearAccumulate, the pixel will be equally smeared
// across histogram bin -1 and histogram bin 0.  The histogram row indices
// for this pixel are irow=-1 row_frac=0.5.

static inline void InitIndices(
    vec_int&    row_indices,    // out
    vec_double& row_fracs,      // out
    vec_int&    col_indices,    // out
    vec_double& col_fracs,      // out
    vec_double& pixelweights,   // out
    const int   patchwidth)     // in: in pixels
{
    CV_Assert(patchwidth % 2 == 1); // patchwidth must be odd in this implementation

    const int npix = SQ(patchwidth); // number of pixels in image patch

    row_indices.resize(npix);
    row_fracs.resize(npix);
    col_indices.resize(npix);
    col_fracs.resize(npix);
    pixelweights.resize(npix);

    const int halfpatchwidth = (patchwidth-1) / 2;

    const double grid_rows_per_img_row = GRIDHEIGHT / (patchwidth-1.);
    const double row_offset = GRIDHEIGHT / 2. - .5; // see header comment

    const double grid_cols_per_img_col = GRIDWIDTH / (patchwidth-1.);
    const double col_offset = GRIDWIDTH / 2. - .5;

    // downweight at border of patch is exp(-1 / (2 * WINDOW_SIGMA))
    const double weight = -1 / (WINDOW_SIGMA * GRIDHEIGHT * GRIDWIDTH );

    int ipix = 0;

    for (double patchrow = -halfpatchwidth; patchrow <= halfpatchwidth; patchrow++)
    {
        const double signed_row = patchrow * grid_rows_per_img_row;
        const double row        = signed_row + row_offset;
        const int irow          = int(floor(row));
        const double row_frac   = row - irow;

        CV_DbgAssert(row >= -.5 && row <= GRIDHEIGHT - .5); // same applies to col below

        for (double patchcol = -halfpatchwidth; patchcol <= halfpatchwidth; patchcol++)
        {
            row_indices[ipix] = irow;
            row_fracs[ipix]   = row_frac;

            const double signed_col = patchcol * grid_cols_per_img_col;
            const double col        = signed_col + col_offset;
            const int icol          = int(floor(col));

            col_indices[ipix] = icol;
            col_fracs[ipix]   = col - icol;

            pixelweights[ipix] = // TODO this weights col and row offsets equally
                exp(weight * (SQ(signed_row) + SQ(signed_col)));

            ipix++;
        }
    }
}

// Init the data that doesn't change unless the image, patch width, or
// GRIDHEIGHT or WIDTH changes (i.e. for Stasm this must be called
// once per pyramid lev).

void Hat::Init_(
    const Image& img,        // in: image scaled to current pyramid level
    const int    patchwidth) // in: patch will be patchwidth x patchwidth pixels
{
    patchwidth_ = patchwidth;

    InitGradMagAndOrientMats(magmat_, orientmat_, img);

    InitIndices(row_indices_, row_fracs_, col_indices_, col_fracs_, pixelweights_,
                patchwidth_);
}

// Calculate the image patch gradient mags and orients.
// Note that the mag for a pixel out of the image boundaries is set
// to 0 and thus contributes nothing later in TrilinearAccumulate.

static void GetMagsAndOrients_GeneralCase(
    vec_double&       mags,         // out
    vec_double&       orients,      // out
    const int         ix,           // in: x coord of center of patch
    const int         iy,           // in: y coord of center of patch
    const int         patchwidth,   // in
    const MAT&        magmat,       // in
    const MAT&        orientmat,    // in
    const vec_double& pixelweights) // in
{
    const int halfpatchwidth = (patchwidth-1) / 2;
    int ipix = 0;
    for (int x = iy - halfpatchwidth; x <= iy + halfpatchwidth; x++)
    {
        const double* const magbuf    = Buf(magmat)    + x * magmat.cols;
        const double* const orientbuf = Buf(orientmat) + x * orientmat.cols;

        for (int y = ix - halfpatchwidth; y <= ix + halfpatchwidth; y++)
        {
            if (x < 0 || x >= magmat.rows || y < 0 || y >= magmat.cols)
            {
                mags[ipix] = 0;    // off image
                orients[ipix] = 0;
            }
            else                   // in image
            {
                mags[ipix]    = pixelweights[ipix] * magbuf[y];
                orients[ipix] = orientbuf[y];
            }
            ipix++;
        }
    }
    CV_DbgAssert(ipix == NSIZE(mags));
}

// Calculate the image patch gradient mags and orients for
// an image patch that is entirely in the image boundaries.

static inline void GetMagsAndOrients_AllInImg(
    vec_double&       mags,                    // out
    vec_double&       orients,                 // out
    const int         ix,                      // in: x coord of center of patch
    const int         iy,                      // in: y coord of center of patch
    const int         patchwidth,              // in
    const MAT&        magmat,                  // in
    const MAT&        orientmat,               // in
    const vec_double& pixelweights)            // in
{
    const int halfpatchwidth = (patchwidth-1) / 2;
    int ipix = 0;
    for (int x = iy - halfpatchwidth; x <= iy + halfpatchwidth; x++)
    {
        const double* const magbuf    = Buf(magmat)    + x * magmat.cols;
        const double* const orientbuf = Buf(orientmat) + x * orientmat.cols;

        for (int y = ix - halfpatchwidth; y <= ix + halfpatchwidth; y++)
        {
            mags[ipix]    = pixelweights[ipix] * magbuf[y];
            orients[ipix] = orientbuf[y];
            ipix++;
        }
    }
    CV_DbgAssert(ipix == NSIZE(mags));
}

void GetMagsAndOrients( // get mags and orients for patch at ix,iy
    vec_double&       mags,         // out
    vec_double&       orients,      // out
    const int         ix,           // in: x coord of center of patch (may be off image)
    const int         iy,           // in: y coord of center of patch (may be off image)
    const int         patchwidth,   // in: in pixels
    const MAT&        magmat,       // in
    const MAT&        orientmat,    // in
    const vec_double& pixelweights) // in
{
    CV_Assert(patchwidth % 2 == 1);  // patchwidth must be odd in this implementation
    const int npix = SQ(patchwidth); // number of pixels in image patch
    const int halfpatchwidth = (patchwidth-1) / 2;

    mags.resize(npix);
    orients.resize(npix);

    if (ix - halfpatchwidth < 0 || ix + halfpatchwidth >= magmat.cols ||
        iy - halfpatchwidth < 0 || iy + halfpatchwidth >= magmat.rows)
    {
        // Part or all of the patch is out the image area.

        GetMagsAndOrients_GeneralCase(mags, orients,
            ix, iy, patchwidth, magmat, orientmat, pixelweights);
    }
    else
    {
        // Patch is entirely in the image area.  The following function returns
        // results identical to GetMagsAndOrients_GeneralCase, but is faster
        // because it doesn't have to worry about the edges of the image.

        GetMagsAndOrients_AllInImg(mags, orients,
            ix, iy, patchwidth, magmat, orientmat, pixelweights);
    }
}

// Apportion the gradient magnitude of a pixel across 8 orientation bins.
// "Accumulate" is in the func name because we "+=" the interpolated values.
// This routine needs to be fast.

static inline void TrilinearAccumulate(
    double& b000, double& b001, // io: histogram bins
    double& b010, double& b011, // io
    double& b100, double& b101, // io
    double& b110, double& b111, // io
    const double mag,           // in: the mag that gets apportioned
    const double rowfrac,       // in
    const double colfrac,       // in
    const double orientfrac)    // in
{
    const double
        a1   = mag * rowfrac,  a0   = mag - a1,

        a11  = a1 * colfrac,   a10  = a1  - a11,
        a01  = a0 * colfrac,   a00  = a0  - a01,

        a111 = a11 * orientfrac,
        a101 = a10 * orientfrac,
        a011 = a01 * orientfrac,
        a001 = a00 * orientfrac;

    b000 += a00 - a001; b001 += a001;
    b010 += a01 - a011; b011 += a011;
    b100 += a10 - a101; b101 += a101;
    b110 += a11 - a111; b111 += a111;
}

// The dimension of histbins is 1+GRIDHEIGHT+1 by 1+GRIDWIDTH+1 by BINS_PER_HIST+1.
// The extra bins are for fast trilinear accumulation (boundary checks unneeded).
// The final bin in each histogram is for degrees greater than 360, needed as
// degrees less than but near 360 get smeared out by trilinear interpolation.

static inline int HistIndex(int row, int col, int iorient) // index into histbins
{
    return ((row+1) * (1+GRIDWIDTH+1) + (col+1)) * (BINS_PER_HIST+1) + iorient;
}

void GetHistograms(                // get all histogram bins
    vec_double&       histbins,    // out
    const int         patchwidth,  // in: in pixels
    const vec_double& mags,        // in
    const vec_double& orients,     // in
    const vec_int&    row_indices, // in
    const vec_double& row_fracs,   // in
    const vec_int&    col_indices, // in
    const vec_double& col_fracs)   // in
{
    const int npix = SQ(patchwidth); // number of pixels in image patch

    const int nhistbins =
        (1 + GRIDHEIGHT + 1) * (1 + GRIDWIDTH + 1) * (BINS_PER_HIST + 1);

    // resize and clear (the fill is needed if the size of histbins
    // doesn't change, because in that case resize does nothing)
    histbins.resize(nhistbins);
    fill(histbins.begin(), histbins.end(), 0.);

    for (int ipix = 0; ipix < npix; ipix++)
    {
        const double orient = orients[ipix];
        const int iorient   = int(floor(orient));
        CV_DbgAssert(iorient >= 0 && iorient < BINS_PER_HIST);

        const int ibin =
            HistIndex(row_indices[ipix], col_indices[ipix], iorient);

        double* const p = &histbins[ibin];

        TrilinearAccumulate( // apportion grad mag across eight orientation bins
            p[0],                                     // ThisOrient
            p[1],                                     // NextOrient
            p[BINS_PER_HIST + 1],                     // NextCol ThisOrient
            p[BINS_PER_HIST + 2],                     // NextCol NextOrient
            p[(GRIDWIDTH+2) * (BINS_PER_HIST+1)],     // NextRow ThisOrient
            p[(GRIDWIDTH+2) * (BINS_PER_HIST+1) + 1], // NextRow NextOrient
            p[(GRIDWIDTH+3) * (BINS_PER_HIST+1)],     // NextRow NextCol ThisOrient
            p[(GRIDWIDTH+3) * (BINS_PER_HIST+1) + 1], // NextRow NextCol NextOrient
            mags[ipix],        // the mag that gets apportioned
            row_fracs[ipix],   // rowfrac
            col_fracs[ipix],   // colfrac
            orient - iorient); // orientfrac
    }
}

static void WrapHistograms(
    vec_double& histbins)
{
    for (int row = 0; row < GRIDHEIGHT; row++)
        for (int col = 0; col < GRIDWIDTH; col++)
        {
            const int ibin = HistIndex(row, col, 0);
            histbins[ibin] += histbins[ibin + BINS_PER_HIST]; // 360 wraps to 0
        }
}

static void CopyHistsToDesc(    // copy histograms to descriptor, skipping pad bins
    VEC&       desc,            // out
    const vec_double& histbins) // in
{
    for (int row = 0; row < GRIDHEIGHT; row++)
        for (int col = 0; col < GRIDWIDTH; col++)
            memcpy(Buf(desc) +
                       (row * GRIDWIDTH + col) * BINS_PER_HIST,
                   &histbins[HistIndex(row, col, 0)],
                   BINS_PER_HIST * sizeof(histbins[0]));
}

static void NormalizeDesc( // take sqrt of elems and divide by L2 norm
    VEC& desc)             // io
{
    double* const data = Buf(desc);
    for (int i = 0; i < NSIZE(desc); i++)
        data[i] = sqrt(data[i]); // sqrt reduces effect of outliers
    const double norm = cv::norm(desc); // L2 norm
    if (!IsZero(norm))
    {
        const double scale = FINAL_SCALE / norm;
        for (int i = 0; i < NSIZE(desc); i++)
            data[i] *= scale;
    }
}

// Hat::Init_ must be called before calling this function.
//
// A HAT descriptor is a vector of doubles of length
// GRIDHEIGHT * GRIDWIDTH * BINS_PER_HIST (currently 4 * 5 * 8 = 160).
//
// The descriptor is a vector of doubles (instead of say bytes) primarily
// so the HatFit function we apply later is fast (because byte-to-double
// type conversions are unneeded when applying the formula).
//
// Note also that a trial implementation that used floats instead of
// doubles (and with a float form of HatFit) was slower.

VEC Hat::Desc_( // return HAT descriptor, Init_ must be called first
    const double x,    // in: x coord of center of patch (may be off image)
    const double y)    // in: y coord of center of patch (may be off image)
    const
{
    CV_Assert(magmat_.rows);         // verify that Hat::Init_ was called

#if _OPENMP // can't be static because multiple instances
    vec_double        mags, orients;
    vec_double        histbins;
#else       // static faster since size rarely changes
    vec_double mags, orients; // the image patch grad mags and orientations
    vec_double histbins;      // the histograms
#endif

    GetMagsAndOrients(mags, orients,
                      cvRound(x), cvRound(y), patchwidth_,
                      magmat_, orientmat_, pixelweights_);

    GetHistograms(histbins,
                  patchwidth_, mags, orients,
                  row_indices_, row_fracs_, col_indices_, col_fracs_);

    WrapHistograms(histbins);        // wrap 360 degrees back to 0

    VEC desc(GRIDHEIGHT * GRIDWIDTH * BINS_PER_HIST, 1); // the HAT descriptor

    CopyHistsToDesc(desc,
                    histbins);

    NormalizeDesc(desc);

    return desc;
}

} // namespace stasm
