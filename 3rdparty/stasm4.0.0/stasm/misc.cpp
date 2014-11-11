// misc.h: miscellaneous definitions for Stasm
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"
#include <sys/stat.h>

namespace stasm
{
//-----------------------------------------------------------------------------

// Like sprintf but returns the string and so doesn't require a buffer arg.

const char* ssprintf(const char* format, ...)
{
    static char s[SBIG];
    va_list args;
    va_start(args, format);
    VSPRINTF(s, format, args);
    va_end(args);
    return s;
}

// Like strncpy but always zero terminate, issue error if can't.

void strncpy_(
    char*       dest, // out
    const char* src,  // in
    int         n)    // in: sizeof(dest)
{
    const char* const start = src; // used only for possible error message
    while ((*dest++ = *src++))     // assignment is intentional
        if (--n <= 0)
            Err("Error in strncpy_ %-80s", start);
}

void ToLowerCase(
    char* s)      // io: convert to lower case
{
    for (; *s; s++)
        *s = char(tolower(*s));
}

void ConvertBackslashesToForwardAndStripFinalSlash(char* s)
{
    int i;

    for (i = 0; s[i]; i++)       // convert \ to /
        if (s[i] == '\\')
            s[i] = '/';

    if (i > 0 && s[i-1] == '/')  // remove final / if any
        s[i-1] = 0;
}

// Get basename and extension e.g. given "C:/bin/cat.exe" returns "cat.exe".

const char* BaseExt(const char* path)
{
    static char s[SLEN];
    char base[SLEN], ext[SLEN];
    splitpath(path, NULL, NULL, base, ext);
    sprintf(s, "%s%s", base, ext);
    return s;
}

// Get basename e.g. given "C:/bin/cat.exe" returns "cat".

const char* Base(const char* path)
{
    static char s[SLEN];
    splitpath(path, NULL, NULL, s, NULL);
    return s;
}

// Our own version of splitpath so we don't need the WIN32 code under Unix.
// This has not been tested for every possible combination but seems to work.

void splitpath(
    const char* path,  // in
    char*       drive, // out: can be NULL
    char*       dir,   // out: can be NULL
    char*       base,  // out: can be NULL
    char*       ext)   // out: can be NULL, includes dot
{
    CV_Assert(path && STRNLEN(path, _MAX_PATH) < _MAX_PATH);

    if (drive)
    {
        *drive = 0;
        if (*path && *(path+1) == ':')       // has drive prefix?
        {
            *drive++ = *path++;              // copy to drive
            *drive++ = *path++;
            *drive = 0;
        }
    }
    const char* end;
    for (end = path; *end; end++)            // end of path
        ;

    const char* start;
    for (start = end; start != path; )       // start of extension
    {
        start--;
        if (*start == '/' || *start == '\\')
            break;
        if (*start == '.')
        {
            end = start;
            break;
        }
    }
    for (start = end; start != path; )       // start of directory
    {
        start--;
        if (*start == '/' || *start == '\\')
        {
            start++;
            break;
        }
    }
    const char* p;
    if (dir)                                 // copy directory to dir
    {
        for (p = path; p != start; )
            *dir++ = *p++;
        // remove trailing / if any, but keep if just a single / or double //
        if (p > path+1 && *(dir-2) != *(dir-1) &&
           (*(dir-1) == '/' || *(dir-1) == '\\'))
        {
            dir--;
        }
        *dir = 0;
    }
    if (base)                                // copy basename to base
    {
        for (p = start; p != end; )
            *base++ = *p++;
        *base = 0;
    }
    if (ext)                                 // copy extension to ext
    {
        for (p = end; *p; )
            *ext++ = *p++;
        *ext = 0;
    }
    // Check for buffer overflow.  TODO Do this properly (i.e. in loops above).
    CV_Assert(drive == NULL || STRNLEN(drive, _MAX_DRIVE) < _MAX_DRIVE);
    CV_Assert(dir   == NULL || STRNLEN(dir,   _MAX_DIR)   < _MAX_DIR);
    CV_Assert(base  == NULL || STRNLEN(base,  _MAX_FNAME) < _MAX_FNAME);
    CV_Assert(ext   == NULL || STRNLEN(ext,   _MAX_EXT)   < _MAX_EXT);
}

// Our own version of makepath so we don't need the WIN32 code under Unix.
// This has not been tested for every possible combination but seems to work.

void makepath(
    char*       path,  // out
    const char* drive, // in: can be NULL, will append ":" if necessary
    const char* dir,   // in: can be NULL, will append "/" if necessary
    const char* base,  // in: can be NULL,
    const char* ext)   // in: can be NULL, will prepend "." if necessary
{
    CV_Assert(path);

    char* p = path;
    if (drive && *drive)
    {
        *p++ = *drive;
        *p++ = ':';
    }
    if (dir && *dir)
    {
        strncpy_(p, dir, _MAX_DIR);
        p += STRNLEN(dir, _MAX_DIR);
        if (*(p-1) != '/' && *(p-1) != '\\')
            *p++ = '/';
    }
    if (base && *base)
    {
        strncpy_(p, base, _MAX_FNAME);
        p += STRNLEN(base, _MAX_FNAME);
    }
    if (ext && *ext)
    {
        if (*ext != '.')
            *p++ = '.';
        strncpy_(p, ext, _MAX_EXT);
        p += STRNLEN(ext, _MAX_EXT);
    }
    *p = 0;
}

void LogShape( // print mat to log file, this is mostly for debugging and testing
    const MAT&  mat,  // in
    const char* matname) // in
{
    // print in shapefile format
    logprintf("\n00000000 %s\n{ %d %d\n", Base(matname), mat.rows, mat.cols);
    for (int row = 0; row < mat.rows; row++)
    {
        for (int col = 0; col < mat.cols; col++)
        {
            if (int(mat(row, col)) == mat(row, col))
                logprintf("%.0f", mat(row, col));
            else
                logprintf("%.1f", mat(row, col));
            if (col < mat.cols-1)
                logprintf(" ");
        }
        logprintf("\n");
    }
    logprintf("}\n");
}

// This redimensions a matrix and preserves as much of the old data as possible.
// If new matrix is bigger than or same size as the old matrix then all the data
// will be preserved. Unused entries in the new matrix are cleared i.e. set to 0.
// The returned matrix may or may not use the same buffer as mat.

MAT DimKeep(const MAT& mat, int nrows, int ncols)
{
    if (mat.rows == nrows && mat.cols == ncols) // no change needed?
        return mat;
    if (mat.rows * mat.cols == nrows * ncols)   // same number of elements?
    {
        CV_Assert(mat.isContinuous());
        MAT newmat(mat);
        newmat.rows = nrows;
        newmat.cols = ncols;
        newmat.step = ncols * sizeof(newmat(0));
        return newmat;
    }
    // copy as much of the data as will fit in the new matrix
    MAT newmat(nrows, ncols, 0.);
    int minrows = MIN(nrows, mat.rows);
    for (int i = 0; i < minrows; i++)
    {
        const double* const rowbuf = mat.ptr<double>(i);
        double* const rowbuf1 = newmat.ptr<double>(i);
        for (int j = 0; j < ncols; j++)
            rowbuf1[j] = rowbuf[j];
    }
    return newmat;
}

const MAT ArrayAsMat(    // create a MAT from a C array of doubles
    int           nrows, // in
    int           ncols, // in
    const double* data)  // in: array of doubles
{
    // <double *> cast is necessary because OpenCV mat constructors
    // don't know how to use <const double *> (they should?)

    return cv::Mat(nrows, ncols, CV_64FC1, const_cast<double*>(data));
}

void RoundMat( // round mat entries to integers
    MAT& mat)  // io
{
    for (int i = 0; i < mat.rows; i++)
    {
        double* const rowbuf = mat.ptr<double>(i);
        for (int j = 0; j < mat.cols; j++)
            rowbuf[j] = cvRound(rowbuf[j]);
    }
}

// Force pinned landmarks in shape to their pinned position.
// This also returns the mean distance from the output shape to pinnedshape.

double ForcePinnedPoints(
    Shape&      shape,       // io
    const Shape pinnedshape) // in: points that are not pinned have coords 0,0
{
    CV_Assert(pinnedshape.rows >= shape.rows);
    double dist = 0;
    int npinned = 0;
    for (int i = 0; i < shape.rows; i++)
    {
        if (PointUsed(pinnedshape, i)) // pinned landmark?
        {
            npinned++;
            dist += PointDist(shape, pinnedshape, i);
            shape(i, IX) = pinnedshape(i, IX);
            shape(i, IY) = pinnedshape(i, IY);
        }
    }
    CV_Assert(npinned > 0);
    return dist / npinned;
}

void ShapeMinMax(
    double&      xmin,  // out
    double&      xmax,  // out
    double&      ymin,  // out
    double&      ymax,  // out
    const Shape& shape) // in
{
    xmin = FLT_MAX, xmax = -FLT_MAX, ymin = FLT_MAX, ymax = -FLT_MAX;
    for (int i = 0; i < shape.rows; i++)
    {
        double x = shape(i, IX), y = shape(i, IY);
        if (PointUsed(x, y))
        {
            if (x < xmin) xmin = x;
            if (x > xmax) xmax = x;
            if (y < ymin) ymin = y;
            if (y > ymax) ymax = y;
        }
    }
    CV_Assert(xmin < FLT_MAX);
    CV_Assert(xmin < xmax);    // need at least two discrete points in shape
}

double ShapeWidth(const Shape& shape) // width of shape in pixels
{
    CV_Assert(shape.rows > 1);
    double xmin, xmax, ymin, ymax;
    ShapeMinMax(xmin, xmax, ymin, ymax, shape);
    return ABS(xmax - xmin);
}

double ShapeHeight(const Shape& shape) // height of shape in pixels
{
    double xmin, xmax, ymin, ymax;
    ShapeMinMax(xmin, xmax, ymin, ymax, shape);
    return ABS(ymax - ymin);
}

// Jitter points at 0,0 if any.  We do this because if both x and y coords
// of a point are zero, Stasm takes that to mean that the point is unused.
// So prevent that when we know all points in the shape are actually used.

Shape JitterPointsAt00(
    const Shape& shape) // in
{
    Shape outshape(shape.clone());

    for (int i = 0; i < outshape.rows; i++)
        if (!PointUsed(outshape, i))
            outshape(i, IX) = XJITTER;

    return outshape;
}

// Multiply a two element xy vector by a 3 x 3 matrix.
// Used for homogeneous transforms.  mat can be 3x2 or 2x2 (since the
// bottom row of a homogeneous mat is constant and is ignored here).

static void Mat33TimesVec(
    VEC&       v,   // io: two element vector
    const MAT& mat) // in: three column matrix
{
    CV_DbgAssert(v.rows == 1 && v.cols == 2);
    CV_DbgAssert(mat.rows >= 2 && mat.cols == 3);
    CV_Assert(mat.isContinuous());

    const double* const data = Buf(mat);

    const double x = v(0, 0);
    const double y = v(0, 1);

    v(0, 0) = data[0] * x + data[1] * y + data[2];
    v(0, 1) = data[3] * x + data[4] * y + data[5];
}

// Transform shape by multiplying it by a homogeneous alignment_mat.
// alignment_mat can be 3x2 or 2x2 (since the bottom row of a homogeneous mat
// is constant and is ignored here).

void AlignShapeInPlace(
    Shape&     shape,         // io
    const MAT& alignment_mat) // in
{
    CV_Assert(shape.cols == 2);
    CV_Assert(alignment_mat.cols == 3 || alignment_mat.rows == 2);

    for (int i = 0; i < shape.rows; i++)
        if (PointUsed(shape, i))
        {
            VEC row(shape.row(i));
            Mat33TimesVec(row, alignment_mat);
            // if transformed point happens to be at 0,0, jitter it
            if (!PointUsed(shape, i))
                shape(i, IX) = XJITTER;
        }
}

void AlignShapeInPlace(
    Shape& shape,                    // io
    double x0, double y0, double z0, // in
    double x1, double y1, double z1) // in
{
    double transform_data[] =
    {
        x0, y0, z0,
        x1, y1, z1
    };
    AlignShapeInPlace(shape, MAT(2, 3, transform_data));
}

Shape AlignShape(                    // return transformed shape
    const Shape& shape,              // in
    const MAT&   alignment_mat)      // in
{
    Shape outshape(shape.clone());
    AlignShapeInPlace(outshape, alignment_mat);
    return outshape;
}

Shape AlignShape(                    // return transformed shape
    const Shape& shape,              // in
    double x0, double y0, double z0, // in
    double x1, double y1, double z1) // in
{
    Shape outshape(shape.clone());
    AlignShapeInPlace(outshape, x0, y0, z0, x1, y1, z1);
    return outshape;
}

// Solves Ax=b by LU decomposition.  Returns col vec x.
// The b argument must be a vector (row or column, it doesn't matter).
// If mat is singular this will fail.

static const VEC Solve(MAT& mat, VEC& b) // note that mat and b get destroyed
{
    CV_Assert(mat.isContinuous() && b.isContinuous());

    if (!cv::LU(Buf(mat), mat.cols * sizeof(mat(0)), mat.rows,
                Buf(b), sizeof(mat(0)), 1))
        Err("Solve: LU failed");

    return b;
}

// Return the similarity transform to align shape to to anchorshape.
// This returns the transformation matrix i.e. the pose.
//
// This is a similarity transform (translation, scaling, and rotation
// but no shearing).  The transform matrix has the form
//
//      a -b  tx
//      b  a  ty
//      0  0   1
//
// See algorithm C.3 in Appendix C of CootesTaylor 2004
// www.isbe.man.ac.uk/~bim/Mods/app_models.pdf.

const MAT AlignmentMat(
    const Shape&  shape,       // in
    const Shape&  anchorshape, // in
    const double* weights)     // in: if NULL (default) all points equally weighted
{
    double W = 0;
    double sx = 0, sy = 0, sx1 = 0, sy1 = 0;
    double sxx_syy = 0, sxx1_syy1 = 0, sxy1_syx1 = 0;

    for (int i = 0; i < shape.rows; i++)
    {
        const double x  = shape(i, IX);
        const double y  = shape(i, IY);
        const double x1 = anchorshape(i, IX);
        const double y1 = anchorshape(i, IY);
        if (PointUsed(x, y) && PointUsed(x1, y1))
        {
            const double w = (weights? weights[i]: 1.);
            W   += w;
            sx  += w * x;
            sy  += w * y;
            sx1 += w * x1;
            sy1 += w * y1;
            sxy1_syx1 += w * (x * y1 - y * x1);
            sxx1_syy1 += w * (x * x1 + y * y1);
            sxx_syy   += w * (x * x  + y * y);
        }
    }
    MAT soln_data = (MAT(4,4) <<  sxx_syy,       0,  sx,  sy,
                                        0, sxx_syy, -sy,  sx,
                                       sx,     -sy,   W,   0,
                                       sy,      sx,   0,   W );

    VEC vec_data = (MAT(4,1) << sxx1_syy1,
                                sxy1_syx1,
                                      sx1,
                                      sy1 );

    const VEC soln(Solve(soln_data, vec_data));

    return (MAT(3, 3) << soln(0), -soln(1), soln(2),    //  a -b tx
                         soln(1),  soln(0), soln(3),    //  b  a ty
                               0,        0,       1 );  //  0  0  1
}

static CvScalar ToCvColor(unsigned color)
{
    CvScalar cvcolor;
    cvcolor.val[0] = (color         & 0xff);
    cvcolor.val[1] = ((color >> 8)  & 0xff);
    cvcolor.val[2] = ((color >> 16) & 0xff);
    cvcolor.val[3] = 0;
    return cvcolor;
}

void DrawShape(             // draw a shape on an image
    CImage&      img,       // io
    const Shape& shape,     // in
    unsigned     color,     // in: rrggbb, default is 0xff0000 (red)
    bool         dots,      // in: true for dots only, default is false
    int          linewidth) // in: default -1 means automatic
{
    const double width = ShapeWidth(shape);
    if (linewidth <= 0)
        linewidth = width > 700? 3: width > 300? 2: 1;
    CvScalar cvcolor(ToCvColor(color));
    int i = 0, j=0;
    do  // use do and not for loop because some points may be unused
    {
        while (i < shape.rows && !PointUsed(shape, i)) // skip unused points
            i++;
        if (i < shape.rows)
        {
            if (dots)
            {
                const int ix = cvRound(shape(i, IX)), iy = cvRound(shape(i, IY));
                if (ix >= 0 && ix < img.cols && iy >= 0 && iy < img.rows)
                {
                    img(iy, ix)[0] = (color >>  0) & 0xff;
                    img(iy, ix)[1] = (color >>  8) & 0xff;
                    img(iy, ix)[2] = (color >> 16) & 0xff;
                }
            }
            else // lines
            {
                j = i+1;
                while (j < shape.rows && !PointUsed(shape, j))
                    j++;
                if (j < shape.rows)
                    cv::line(img,
                             cv::Point(cvRound(shape(i, IX)), cvRound(shape(i, IY))),
                             cv::Point(cvRound(shape(j, IX)), cvRound(shape(j, IY))),
                             cvcolor, linewidth);
            }
        }
        i++;
    }
    while (i != shape.rows && j != shape.rows);
}

void ImgPrintf(         // printf on image
    CImage&     img,    // io
    double      x,      // in
    double      y,      // in
    unsigned    color,  // in: rrggbb e.g. 0xff0000 is red
    double      size,   // in: relative font size, 1 is standard size
    const char* format, // in
                ...)    // in
{
    char s[SBIG];       // format format into s
    va_list args;
    va_start(args, format);
    VSPRINTF(s, format, args);
    va_end(args);

    CV_Assert(size > 0);
    double fontsize = size * MIN(img.cols, img.rows) / 1000.;
    if (fontsize < .3) // smaller than about .3 is not legible
        fontsize = .3;

    // make the letters thick enough to be seen on high pixel images,
    // but not too thick to be illegible.  The code below sorta works.

    int thickness = MAX(1, cvRound(img.rows > 1000? 2 * fontsize: fontsize));

    putText(img, s, cv::Point(cvRound(x), cvRound(y)),
            CV_FONT_HERSHEY_SIMPLEX, fontsize, ToCvColor(color), thickness);
}

static byte RgbToGray( // CIE conversion to gray using integer arithmetic
    const RGBV rgb)
{
    return byte((2990 * rgb[2] + 5870 * rgb[1] + 1140 * rgb[0] + 5000) / 10000);
}

void DesaturateImg( // for apps and debugging, unneeded for ASM
    CImage& img)    // io: convert to gray (but still an RGB image)
{
    for (int i = 0; i < img.rows; i++)
    {
        RGBV* const rowbuf = img.ptr<RGBV>(i);
        for (int j = 0; j < img.cols; j++)
        {
            byte * const p = (byte *)(rowbuf + j);
            p[0] = p[1] = p[2] = RgbToGray(rowbuf[j]);
        }
    }
}

void ForceRectIntoImg(   // force rectangle into image
    int&         ix,     // io
    int&         iy,     // io
    int&         ncols,  // io
    int&         nrows,  // io
    const Image& img)    // in
{
    ix = Clamp(ix, 0, img.cols-1);

    int ix1 = ix + ncols;
    if (ix1 > img.cols)
        ix1 = img.cols;

    ncols = ix1 - ix;

    CV_Assert(ix >= 0 && ix < img.cols);
    CV_Assert(ix + ncols  >= 0 && ix + ncols  <= img.cols);

    iy = Clamp(iy, 0, img.rows-1);

    int iy1 = iy + nrows;
    if (iy1 > img.rows)
        iy1 = img.rows;

    nrows = iy1 - iy;

    CV_Assert(iy >= 0 && iy < img.rows);
    CV_Assert(iy + nrows >= 0 && iy + nrows <= img.rows);
}

void ForceRectIntoImg(   // force rectangle into image
    Rect&        rect,   // io
    const Image& img)    // in
{
    ForceRectIntoImg(rect.x, rect.y, rect.width, rect.height, img);
}

Image FlipImg(const Image& img) // in: flip image horizontally (mirror image)
{
    Image workimg(img.isContinuous()? img: img.clone()); // need continuous image
    const int width = workimg.cols;
    const int height = workimg.rows;
    Image outimg(height, width);
    for (int iy = 0; iy < height; iy++)
    {
        int width1 = iy * width;
        int ix1 = width;
        for (int ix = 0; ix < width; ix++)
            outimg.data[ix + width1] = workimg.data[--ix1 + width1];
    }
    return outimg;
}

void FlipImgInPlace(Image& img) // io: flip image horizontally (mirror image)
{
    img = FlipImg(img);
}

void OpenDetector( // open face or feature detector from its XML file
    cv::CascadeClassifier& cascade,  // out
    const char*            filename, // in: basename.ext of cascade
    const char*            datadir)  // in
{
    if (cascade.empty()) // not yet opened?
    {
        char dir[SLEN]; STRCPY(dir, datadir);
        ConvertBackslashesToForwardAndStripFinalSlash(dir);

        char path[SLEN]; sprintf(path, "%s/%s", dir, filename);

        logprintf("Open %s\n", path);

        if (!cascade.load(path))
            Err("Cannot load %s", path);
    }
}

// convert the x and y coords in feats from the search ROI to the image frame

static void DiscountSearchRegion(
    vec_Rect& feats,              // io
    Rect&     searchrect)         // in
{
    for (int ifeat = 0; ifeat < NSIZE(feats); ifeat++)
    {
        feats[ifeat].x += searchrect.x;
        feats[ifeat].y += searchrect.y;
    }
}

vec_Rect Detect(                            // detect faces or facial features
    const Image&           img,             // in
    cv::CascadeClassifier* cascade,         // in
    const Rect*            searchrect,      // in: search in this region, can be NULL
    double                 scale_factor,    // in
    int                    min_neighbors,   // in
    int                    flags,           // in
    int                    minwidth_pixels) // in: reduces false positives
{
    CV_Assert(!cascade->empty());

    Rect searchrect1; searchrect1.width = 0;
    if (searchrect)
    {
        searchrect1 = *searchrect;
        ForceRectIntoImg(searchrect1, img);
        if (searchrect1.height == 0)
            searchrect1.width = 0;
    }
    Image roi(img,
              searchrect1.width? searchrect1: Rect(0, 0, img.cols, img.rows));

    // TODO If we don't allocate feats now we get a crash on mem release later.

    const int MAX_NFACES_IN_IMG = int(1e4); // arb, but big
    vec_Rect feats(MAX_NFACES_IN_IMG);

    // Note: This call to detectMultiScale causes the Peak Working Set
    // to jump to 160 MBytes (multiface2.jpg) versus less than 50 MBytes
    // for the rest of Stasm (Feb 2013).

    cascade->detectMultiScale(roi, feats, scale_factor, min_neighbors, flags,
                              cvSize(minwidth_pixels, minwidth_pixels));

    if (!feats.empty() && searchrect1.width)
        DiscountSearchRegion(feats, searchrect1);

    return feats;
}

bool IsLeftFacing(EYAW eyaw) // true if eyaw is for a left facing face
{
    return int(eyaw) <= int(EYAW_22);
}

int EyawAsModIndex(      // note: returns a negative index for left facing yaws
    EYAW           eyaw, // in
    const vec_Mod& mods) // in: a vector of models, one for each yaw range
{
    int imod = 0;
    if (NSIZE(mods) > 1)
    {
        switch (eyaw)
        {
            case EYAW00:  imod =  0; break;
            case EYAW_45: imod = -2; break;
            case EYAW_22: imod = -1; break;
            case EYAW22:  imod =  1; break;
            case EYAW45:  imod =  2; break;
            default:      Err("EyawAsModIndex: bad eyaw %d", eyaw); break;
        }
    }
    CV_Assert(ABS(imod) < NSIZE(mods));
    return imod;
}

EYAW DegreesAsEyaw( // this determines what model is best for a given yaw
    double yaw,     // in: yaw in degrees, negative if left facing
    int    nmods)   // in
{
    (void) yaw;
    (void) nmods;

    if (nmods == 1)
        return EYAW00;

#if MOD_3 || MOD_A || MOD_A_EMU // experimental versions
    if (yaw < -EYAW_TO_USE_DET45)
        return EYAW_45;

    else if (yaw < -EYAW_TO_USE_DET22)
        return EYAW_22;

    else if (yaw <= EYAW_TO_USE_DET22)
        return EYAW00;

    else if (yaw <= EYAW_TO_USE_DET45)
        return EYAW22;

    return EYAW45;
#else
    CV_Assert(0);
    return EYAW00; // keep compiler quiet
#endif
}

const char* EyawAsString(EYAW eyaw) // utility for debugging/tracing
{
    switch (int(eyaw))
    {
        case EYAW00:  return "YAW00";
        case EYAW_45: return "YAW_45";
        case EYAW_22: return "YAW_22";
        case EYAW22:  return "YAW22";
        case EYAW45:  return "YAW45";
        case INVALID: return "YAW_Inv";
        default:      Err("YawAsString: Invalid eyaw %d", eyaw); break;
    }
    return NULL; // prevent compiler warning
}

DetPar FlipDetPar(          // mirror image of detpar
    const DetPar& detpar,   // in
    int           imgwidth) // in
{
    DetPar detpar_new(detpar);

    detpar_new.x = imgwidth - detpar.x;
    detpar_new.y = detpar.y;
    detpar_new.width  = detpar.width;
    detpar_new.height = detpar.height;
    const bool valid_leye = Valid(detpar.lex);
    const bool valid_reye = Valid(detpar.rex);

    detpar_new.lex = detpar_new.ley =
    detpar_new.rex = detpar_new.rey =
    detpar_new.mouthx = detpar_new.mouthy = INVALID;

    if (valid_leye)
    {
        detpar_new.rex = imgwidth - detpar.lex;
        detpar_new.rey = detpar.ley;
    }
    if (valid_reye)
    {
        detpar_new.lex = imgwidth - detpar.rex;
        detpar_new.ley = detpar.rey;
    }
    if (Valid(detpar.mouthx))
    {
        detpar_new.mouthx = imgwidth - detpar.mouthx;
        detpar_new.mouthy = detpar.mouthy;
    }
    return detpar_new;
}

} // namespace stasm
