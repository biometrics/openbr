// misc.h: miscellaneous definitions for Stasm
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_MISC_H
#define STASM_MISC_H

namespace stasm
{
using cv::Rect;
using std::vector;
using std::string;

typedef vector<int>      vec_int;
typedef vector<double>   vec_double;
typedef vector<cv::Rect> vec_Rect;

typedef unsigned char byte;

typedef cv::Mat_<double> MAT;   // a matrix with double elements
typedef cv::Mat_<double> VEC;   // by convention indicates one-dim matrix

typedef cv::Mat_<double> Shape; // by convention an N x 2 matrix holding a shape

typedef cv::Mat_<byte> Image;   // a gray image (a matrix of bytes)

typedef cv::Vec3b RGBV;         // a vec of three bytes: red(0), green, and blue(2)

typedef cv::Mat_<RGBV> CImage;  // an RGB image (for apps and debugging, unneeded for ASM)

static const int IX = 0;        // X,Y index in shape matrices.  For clarity by
static const int IY = 1;        // convention we use these rather than 0 and 1.

static const int SLEN = 260;    // generic string length
                                // big enough for any Windows path (MAX_PATH is 260)

static const int SBIG = 10000;  // long string length, enough for big printfs

#ifndef _MAX_PATH // definitions copied verbatim from Microsoft stdlib.h
#define _MAX_PATH   260 /* max. length of full pathname */
#define _MAX_DRIVE  3   /* max. length of drive component */
#define _MAX_DIR    256 /* max. length of path component */
#define _MAX_FNAME  256 /* max. length of file name component */
#define _MAX_EXT    256 /* max. length of extension component */
#endif

// Secure form of strcpy and friends (prevent buffer overrun).
// The CV_DbgAssert catches an easy programming error where
// we mistakenly take the size of a pointer.

#if _MSC_VER                            // microsoft compiler
  #define STRCPY(dest, src) \
          { \
          CV_DbgAssert(sizeof(dest) > 8); \
          strcpy_s(dest, sizeof(dest), src); \
          }
  #define STRCAT(dest, src) \
          { \
          CV_DbgAssert(sizeof(dest) > 8); \
          strcat_s(dest, sizeof(dest), src); \
          }
  #define VSPRINTF(dest, format, args) \
          { \
          CV_DbgAssert(sizeof(dest) > 8); \
          vsnprintf_s(dest, sizeof(dest), _TRUNCATE, format, args); \
          }
#else
  #define STRCPY(dest, src) \
          { \
          CV_DbgAssert(sizeof(dest) > 8); \
          strncpy_(dest, src, sizeof(dest)); \
          }
  #define STRCAT(dest, src) \
          { \
          CV_DbgAssert(sizeof(dest) > 8); \
          strncat(dest, sizeof(dest), src); \
          }
  #define VSPRINTF(dest, format, args) \
          { \
          CV_DbgAssert(sizeof(dest) > 8); \
          vsnprintf(dest, sizeof(dest), format, args); \
          }
#endif

// A macro to disallow the copy constructor and operator= functions.
// This is used in the private declarations for a class where those member
// functions have not been explicitly defined.  This macro prevents use of
// the implicitly defined functions (the compiler will complain if you try
// to use them).
// This is often just paranoia.  The implicit functions may actually be ok
// for the class in question, but shouldn't be used until that is checked.
// For details, See Item 6 Meyers Effective C++ and the Google C++ Style Guide.

#define DISALLOW_COPY_AND_ASSIGN(ClassName) \
    ClassName(const ClassName&);            \
    void operator=(const ClassName&)

template <typename T> int NELEMS(const T& x)     // number of elems in an array
{
    return int(sizeof(x) / sizeof((x)[0]));
}

// The NSIZE and STRNLEN utility functions prevent the following
// warnings from certain compilers:
//      o  signed/unsigned mismatch
//      o  conversion from 'size_t' to 'int', possible loss of data
// Alternatives would be to use typecasts directly in the code
// or pedantically use size_t instead of int.

static inline int NSIZE(const MAT& m)            // nrows * ncols
{
    return int((m).total());
}

template <typename T> int NSIZE(const T& x)      // size of any STL container
{
    return int(x.size());
}

static inline int STRNLEN(const char* s, int n)
{
    return int(strnlen(s, n));
}

template <typename T> T SQ(const T x)            // define SQ(x)
{
    return x * x;
}

template <typename T> T ABS(const T x)           // define ABS(x)
{
    // portable across compilers unlike "abs"
    return x < 0? -x: x;
}

template <typename T> T Clamp(const T x, const T min, const T max)
{
    return MIN(MAX(x, min), max);
}

// Equal() returns true if x == y within reasonable tolerance.
// The 1e-7 is arbitrary but approximately equals FLT_EPSILON.
// (If one or both of the numbers are NANs then the test fails, even if
// they are equal NANs.  Which is not necessarily desireable behaviour.)

static inline bool Equal(const double x, const double y, const double max = 1e-7)
{
    return ABS(x-y) < max;
}

static inline bool IsZero(const double x, const double max = 1e-7)
{
    return Equal(x, 0, max);
}

static inline double RadsToDegrees(const double rads)
{
    return 180 * rads / 3.14159265358979323846264338328;
}

static const int INVALID = 99999; // used to specify unavail eye locations, etc

template <typename T> bool Valid(const T x)
{
    return x != INVALID && x != -INVALID;
}

// For reference, the fields of an OpenCV Mat are as follows.
// See \OpenCV\build\include\opencv2\core\core.hpp for details.
//
//    int flags;        // magic signature, continuity flag, depth, number of chans
//    int dims;         // matrix dimensionality, >= 2
//    int rows, cols;   // number of rows and columns or (-1, -1)
//    uchar* data;      // the data
//    int* refcount;    // pointer to ref counter, NULL if user-allocated
//    uchar* datastart; // fields used in locateROI and adjustROI
//    uchar* dataend;
//    uchar* datalimit;
//    MatAllocator* allocator; // custom allocator
//    MSize size;
//    MStep step;

static inline double* Buf(const MAT& mat) // access MAT data buffer
{
    return (double*)(mat.data);
}

static inline VEC AsColVec(const MAT& mat) // view entire matrix as a col vector
{
    CV_Assert(mat.isContinuous());
    return MAT(mat.rows * mat.cols, 1, Buf(mat));
}

static inline VEC AsRowVec(const MAT& mat) // view entire matrix as a row vector
{
    CV_Assert(mat.isContinuous());
    return MAT(1, mat.rows * mat.cols, Buf(mat));
}

// Note on unused points:
//
//   Unused points (a.k.a. missing points) points are indicated
//   by setting both x and y equal to zero.
//   Thus if there is a valid point that happens actually to
//   be at 0,0 (rare) we must offset x slightly to ensure that the
//   point is seen by Stasm as used.  Hence XJITTER.
//
//   XJITTER is one tenth of a pixel, which is big enough to be
//   visible when saved in a shapefile with one decimal digit.
//
//   Unused points are mostly useful during training (it is not unusual for a
//   landmark to be obscured in a training face).  They are also used during
//   a search with pinned points (non-pinned points are marked as unused in
//   the shape which specifies the pinned points).

static const double XJITTER = .1;

static inline bool PointUsed(const double x, const double y)
{
    return !IsZero(x, XJITTER) || !IsZero(y, XJITTER);
}

static inline bool PointUsed(const Shape& shape, int ipoint)
{
    return PointUsed(shape(ipoint, IX), shape(ipoint, IY));
}

static inline double PointDist(
    double x1,  // in
    double y1,  // in
    double x2,  // in
    double y2)  // in
{
    CV_Assert(PointUsed(x1, y1));
    CV_Assert(PointUsed(x2, y2));

    return sqrt(SQ(x1 - x2) + SQ(y1 - y2));
}

static inline double PointDist(
    const Shape& shape1,   // in: the first shape
    const Shape& shape2,   // in: the second shape
    int          ipoint)   // in: the point
{
    return PointDist(shape1(ipoint, IX), shape1(ipoint, IY),
                     shape2(ipoint, IX), shape2(ipoint, IY));
}

static inline double PointDist(
    const Shape& shape,    // in
    int          ipoint1,  // in: the first point
    int          ipoint2)  // in: the second point
{
    return PointDist(shape(ipoint1, IX), shape(ipoint1, IY),
                     shape(ipoint2, IX), shape(ipoint2, IY));
}

// note: in frontal-model-only Stasm, the only valid value for EYAW is EYAW00

enum EYAW
{
    EYAW_45 = -3, // yaw -45 degrees (left facing strong three-quarter pose)
    EYAW_22 = -2, // yaw -22 degrees (left facing mild three-quarter pose)
    EYAW00  =  1, // yaw 0 degrees   (frontal pose)
    EYAW22  =  2, // yaw 22 degrees  (right facing mild three-quarter pose)
    EYAW45  =  3  // yaw 45 degrees  (right facing strong three-quarter pose)
};

enum ESTART // do we use the detected eyes or mouth to help position the startshape?
            // note: gaps in enum numbering are for compat with other Stasm versions
{
    ESTART_RECT_ONLY     = 1, // use just the face det rect to align the start shape
    ESTART_EYES          = 2, // use eyes if available (as well as face rect)
    ESTART_EYE_AND_MOUTH = 4  // uses eye(s) and mouth if both available
};

#if MOD_3 || MOD_A || MOD_A_EMU // experimental versions
static double EYAW_TO_USE_DET22 = 14; // what estimated yaw requires the yaw22 mod
static double EYAW_TO_USE_DET45 = 35; // ditto for yaw45 model
#endif

struct DetPar // the structure describing a face detection
{
    double x, y;           // center of detector shape
    double width, height;  // width and height of detector shape
    double lex, ley;       // center of left eye, left and right are wrt the viewer
    double rex, rey;       // ditto for right eye
    double mouthx, mouthy; // center of mouth
    double rot;            // in-plane rotation
    double yaw;            // yaw
    EYAW   eyaw;           // yaw as an enum

    DetPar() // constructor sets all fields to INVALID
        : x(INVALID),
          y(INVALID),
          width(INVALID),
          height(INVALID),
          lex(INVALID),
          ley(INVALID),
          rex(INVALID),
          rey(INVALID),
          mouthx(INVALID),
          mouthy(INVALID),
          rot(INVALID),
          yaw(INVALID),
          eyaw(EYAW(INVALID))
    {
    };

};

//-----------------------------------------------------------------------------

const char* ssprintf(const char* format, ...);
void strncpy_(char* dest, const char* src, int n);
void ToLowerCase(char* s);
void ConvertBackslashesToForwardAndStripFinalSlash(char* s);
const char* Base(const char* path);
const char* BaseExt(const char* path);

void splitpath(
    const char* path,
    char* drive, char* dir, char* base, char* ext);

void makepath(
    char* path,
    const char* drive, const char* dir, const char* base, const char* ext);

void LogShape(const MAT& mat, const char* matname);

MAT DimKeep(const MAT& mat, int nrows, int ncols);

const MAT ArrayAsMat(int ncols, int nrows, const double* data);

void RoundMat(MAT& mat); // round mat entries to integers

Shape JitterPointsAt00(const Shape& shape);

double ForcePinnedPoints( // force pinned landmarks in shape to their pinned posn
    Shape&      shape,        // io
    const Shape pinnedshape); // in

void ShapeMinMax(
    double&      xmin,   // out
    double&      xmax,   // out
    double&      ymin,   // out
    double&      ymax,   // out
    const Shape& shape); // in

double ShapeWidth(const Shape& shape);  // width of shape in pixels

double ShapeHeight(const Shape& shape); // height of shape in pixels

void AlignShapeInPlace(               // affine transform of shape
    Shape&     shape,                 // io
    const MAT& alignment_mat);        // in

void AlignShapeInPlace(               // affine transform of shape
    Shape& shape,                     // io
    double x0, double y0, double z0,  // in
    double x1, double y1, double z1); // in

Shape AlignShape(                     // return transformed shape, affine transform
    const Shape& shape,               // in
    const MAT&   alignment_mat);      // in

Shape AlignShape(                     // return transformed shape, affine transform
    const Shape& shape,               // in
    double x0, double y0, double z0,  // in
    double x1, double y1, double z1); // in

const MAT AlignmentMat(          // return similarity transf to align shape to anchorshape
    const Shape&  shape,         // in
    const Shape&  anchorshape,   // in
    const double* weights=NULL); // in: if NULL (default) all points equal weight

void DrawShape(                   // draw a shape on an image
    CImage&      img,             // io
    const Shape& shape,           // in
    unsigned     color=0xff0000,  // in: rrggbb e.g. 0xff0000 is red
    bool         dots=false,      // in: true for dots only
    int          linewidth=0);    // in: default 0 means automatic

void ImgPrintf(                   // printf on image
    CImage&     img,              // io
    double      ix,               // in
    double      iy,               // in
    unsigned    color,            // in: rrggbb e.g. 0xff0000 is red
    double      size,             // in: relative font size, 1 is standard size
    const char* format,           // in
    ...);                         // in

void DesaturateImg(
    CImage& img);       // io: convert to gray (but still an RGB image)

void ForceRectIntoImg(  // force rectangle into image
    int&         ix,    // io
    int&         iy,    // io
    int&         ncols, // io
    int&         nrows, // io
    const Image& img);  // in

void ForceRectIntoImg(  // force rectangle into image
    Rect&        rect,  // io
    const Image& img);  // in

Image FlipImg(const Image& img); // in: flip image horizontally (mirror image)

void FlipImgInPlace(Image& img); // io: flip image horizontally (mirror image)

void OpenDetector( // open face or feature detector from its XML file
    cv::CascadeClassifier& cascade,  // out
    const char*            filename, // in: basename.ext of cascade
    const char*            datadir); // in

vec_Rect Detect(                             // detect faces or facial features
    const Image&           img,              // in
    cv::CascadeClassifier* cascade,          // in
    const Rect*            searchrect,       // in: search in this region, can be NULL
    double                 scale_factor,     // in
    int                    min_neighbors,    // in
    int                    flags,            // in
    int                    minwidth_pixels); // in: reduces false positives

// TODO Following commented out to avoid circular dependency.
// int EyawAsModIndex(EYAW eyaw, const vec_Mod& mods);

bool IsLeftFacing(EYAW eyaw);   // true if eyaw is for a left facing face

EYAW DegreesAsEyaw( // this determines what model is best for a given yaw
    double yaw,     // in: yaw in degrees, negative if left facing
    int    nmods);  // in

const char* EyawAsString(EYAW eyaw);

DetPar FlipDetPar(           // mirror image of detpar
    const DetPar& detpar,    // in
    int           imgwidth); // in

} // namespace stasm
#endif // STASM_MISC_H
