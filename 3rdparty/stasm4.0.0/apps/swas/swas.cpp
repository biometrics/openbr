// swas.cpp: Run stasm on the images listed in a shapefile.
//           Compare landmark results to manually landmarked points and write
//           results to a "fit file" for post-processing (e.g. with swas.R).
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include <sys/types.h>
#include <sys/stat.h>
#include "stasm.h"
#include "opencv/highgui.h"
#include "fitmeas.h"
#include "fm29.h"
#include "../stasm_regex.h"
#include "../shapefile.h"
#include "../appmisc.h"

#ifdef _MSC_VER // microsoft
#pragma warning(disable: 4996)  // 'strcpy' This function may be unsafe
#endif

namespace stasm
{
static bool crop_g = false;     // -c flag
static bool writeimgs_g = true; // -i flag
static bool fitprefix_g = true; // -f flag
static int  minwidth_g = 25;    // -s flag
static bool quiet_g;            // -q flag

// -q flag, func pointer to lprint or logprintf
static void (*printf_g)(const char *pArgs, ...) = &lprintf;

//-----------------------------------------------------------------------------

bool IsDirWriteable(const char* dir)
{
#if _MSC_VER // microsoft
    struct _stat st;
    if (_stat(dir, &st) == 0)
#else
    struct stat st;
    if (stat(dir, &st) == 0)
#endif
        return (st.st_mode & S_IFDIR) != 0 && (st.st_mode & S_IWRITE) != 0;
    return false;
}

static void OpenFitFile(
    FILE*&      fitfile,   // out: opened fit file
    const char* shapepath, // in
    int         nshapes,   // in
    int         seed,      // in
    const char* sregex,    // in
    int         nchar)     // in
{
    if (!IsDirWriteable("fit"))
        Err("Subdirectory \"fit\" does not exist or is not writeable.\n"
            "Do a \"mkdir fit\" then run swas again.\n");

    char sregex1[SLEN]; sregex1[0] = 0;
    if (sregex && sregex[0])
    {
        sprintf(sregex1, "_REGEX_%s", sregex);
        RegexToLegalFilename(sregex1);
    }

    const char *smod = stasm_VERSION;
    while (*smod && *smod != '_') // skip to first underscore
        smod++;
    if (*smod != '_')             // no underscore?
        smod = "";

    char snshapes[SLEN]; snshapes[0] = 0;
    if (nshapes)
        sprintf(snshapes, "_n%d_seed%d", nshapes, seed);

    char fitpath[SLEN];
    sprintf(fitpath, "fit/%s%s%s%s.fit", Base(shapepath), snshapes, sregex1, smod);
    lprintf("Writing %s%s", fitpath, quiet_g? " ": "\n");
    fitfile = fopen(fitpath, "w");
    if (!fitfile)
        Err("Cannot open %s for writing", fitpath);
    Fprintf(fitfile,
            "%-*s meanfit iworst    me17    fm29 iworstfm29 intereye eyemouth estyaw    "
            "yaw  pitch   roll   poseerr dettime asmtime\n",
            nchar+1, "file");
}

static void ProcessShapefileArg(
    ShapeFile&   sh,             // out
    FILE*&       fitfile,        // out: opened fit file
    int&         argc,           // in
    const char** argv)           // in: file.shape [N [SEED [REGEX]]]
{
    char shapepath[SLEN];
    int  nshapes;
    int  seed;
    char sregex[SBIG];

    if (argc < 1)
        Err("No shapefile argument.  Use swas -? for help.");

    strcpy(shapepath, *argv);
    int n = STRNLEN(shapepath, SLEN);
    if (n < 6 || _stricmp(shapepath + n - 6, ".shape"))
        Err("Invalid shape file name %s (expected a .shape suffix)", shapepath);

    nshapes = 0;
    if (argc > 1)
    {
        argv++;
        argc--;
        if (1 != sscanf(*argv, "%d", &nshapes) || nshapes < 0 || nshapes > 1e6)
           Err("Invalid N argument %s (use 0 for all, else positive integer)", *argv);
    }
    seed = 0;
    if (argc > 1)
    {
        argv++;
        argc--;
        if (1 != sscanf(*argv, "%d", &seed))
           Err("Invalid SEED argument %s "
                "(must be an integer, use 0 for first N shapes)", *argv);
    }
    sregex[0] = 0;
    if (argc > 1)
    {
        argv++;
        argc--;
        if (1 != sscanf(*argv, "%s", sregex))
           Err("Invalid regex argument %s", *argv);
    }
    if (argc != 1)
        Err("Too many arguments, do not know what to do with \"%s\"", *(argv+1));

    sh.Open_(shapepath);
    sh.Subset_(nshapes, seed, sregex);
    OpenFitFile(fitfile, shapepath, nshapes, seed, sregex, sh.nchar_);
}

static void ProcessOptions(
    ShapeFile&   sh,        // out
    FILE*&       fitfile,   // out: opened fit file
    int          argc,      // in
    const char** argv)      // in
{
    static const char* usage =
"swas [FLAGS] file.shape [N [SEED [REGEX]]]\n"
"\n"
"Examples:\n"
"    swas file.shape               (all faces in file.shape)\n"
"    swas file.shape 3             (first 3 faces in file.shape)\n"
"    swas file.shape 3 99          (3 random faces, random seed is 99)\n"
"    swas file.shape 0 0 x         (all faces with \"x\" in their name)\n"
"    swas file.shape 3 99 \"xy|z\"   (3 random faces with \"xy\"  or \"z\" in their name)\n"
"\n"
"Flags:\n"
"    -c  crop output images to face\n"
"    -d  enable debug prints\n"
"    -i  do not write landmarked images (faster)\n"
"    -f  do not prefix image names with fit\n"
"    -s  small faces (min width 10%%, default is 25%%)\n"
"    -?  help\n"
"\n"
"Swas version %s    www.milbo.users.sonic.net/stasm\n"; // %s is stasm_VERSION

    if (argc < 2)
        Err("No shapefile argument.  Use swas -? for help.");
    while (argc-- > 0 && (*++argv)[0] == '-')
    {
        if ((*argv + 1)[1])
            Err("Invalid flag -%s (you cannot combine flags).  Use swas -? for help.",
                *argv + 1);
        switch ((*argv + 1)[0])
        {
            case 'c':
                crop_g = true;
                break;
            case 'd':
                trace_g = true;
                break;
            case 'f':
                fitprefix_g = false;
                break;
            case 'i':
                writeimgs_g = false;
                break;
            case 'q':
                quiet_g = true;
                printf_g = &logprintf;  // print only to the log file
                break;
            case 's':
                minwidth_g = 10;
                break;
            case '?':               // -?
                printf(usage, stasm_VERSION);
                exit(1);
            default:
                Err("Invalid flag -%s.  Use swas -? for help.", *argv + 1);
                break;
        }
    }
    ProcessShapefileArg(sh, fitfile,
                        argc, argv);
}

static void WriteLandmarkedImg(
    const Image& img,      // in: the image
    const char*  newpath,  // in: new image path
    const Shape& shape,    // in: the shape
    unsigned     color,    // in: rrggbb e.g. 0xff0000 is red
    int          iworst,   // in: index of worst fitting point, -1 for none
    bool         dots,     // in: true for dots only, default is false
    const Shape& refshape) // in: the refshape (used only to crop the image)
{
    CImage cimg; cvtColor(img, cimg, CV_GRAY2BGR); // color image

    if (iworst >= 0)      // draw circle at worst fitting point?
        cv::circle(cimg,
                   cv::Point(cvRound(shape(iworst, IX)),
                             cvRound(shape(iworst, IY))),
                   MAX(2, cvRound(ShapeWidth(shape) / 40)),
                   cv::Scalar(0, 0, 255), 2);

    DrawShape(cimg, shape, color, dots);

    if (crop_g)
        CropCimgToShapeWithMargin(cimg, refshape);

    cv::imwrite(newpath, cimg);
}

static void PossiblyWriteImgs(
    const Image& img,          // in
    const char*  imgpath,      // in
    const Shape  &shape,       // in
    const Shape  &refshape,    // in
    double       fm29,         // in
    int          iworst,       // in: index of worst fitting point, -1 for none
    double       me17)         // in
{
    if (writeimgs_g) // -i flag
    {
        if (fm29 > .99)
            fm29 = .99;

        const char* const base = Base(imgpath);
        char newpath[SLEN];

        if (fitprefix_g)
            sprintf(newpath, "%5.3f_%s_ref.bmp", me17, base);
        else
            sprintf(newpath, "%s_ref.bmp", base);
        WriteLandmarkedImg(img, newpath, refshape, 0xffff00, -1, true, refshape); // yellow

        if (fitprefix_g)
            sprintf(newpath, "%5.3f_%s_stasm.bmp", me17, base);
        else
            sprintf(newpath, "%s_stasm.bmp", base);
        WriteLandmarkedImg(img, newpath, shape, 0xff0000, iworst, false, refshape); // red
#if 0 // write me17 results
        if (fitprefix_g)
            sprintf(newpath, "%5.3f_%s_sme17.bmp", me17, base);
        else
            sprintf(newpath, "%s_sme17.bmp", base);
        WriteLandmarkedImg(img, newpath, Shape17(refshape), 0xffff00, -1, true, refshape);
        if (fitprefix_g)
            sprintf(newpath, "%5.3f_%s_sme17ref.bmp", me17, base);
        else
            sprintf(newpath, "%s_sme17ref.bmp", base);
        WriteLandmarkedImg(img, newpath, Shape17(refshape), 0xffff00, -1, false, refshape);
#endif
    }
}

static void ProcessFace(
    const Image&    img,            // in
    const char*     imgpath,        // in
    int             foundface,      // in
    const float*    landmarks,      // in
    float           estyaw,         // in
    double          facedet_time,   // in
    double          asmsearch_time, // in
    const Shape     &refshape,      // in
    FILE*           fitfile,        // in
    const ShapeFile &sh)            // in

{
    double meanfit = NOFIT, me17 = NOFIT, fm29 = NOFIT;
    int iworst = -1, iworstfm29 = -1;
    if (!foundface)
        printf_g("no face\n");
    else
    {
        Shape shape(LandmarksAsShape(landmarks));
        if (trace_g)
            LogShape(shape, imgpath);
        meanfit = MeanFitOverInterEye(iworst, shape, refshape);
        me17 = Me17(shape, refshape);
        if (shape.rows == stasm_NLANDMARKS && refshape.rows == stasm_NLANDMARKS)
            Fm29(fm29, iworstfm29, shape, refshape);
        if (refshape.rows == stasm_NLANDMARKS)
            printf_g("Fit %5.3f\n", fm29);
        else
            printf_g("me17 %5.3f\n", me17);
        PossiblyWriteImgs(img, imgpath, shape, refshape, fm29, iworstfm29, me17);
    }
    if (trace_g)
        lprintf("\n");
    const char* const base = Base(imgpath);
    const MAT pose(sh.Pose_(base));
    Fprintf(fitfile,
        "%-*s%s "
        "%7.5f % 6d %7.5f %7.5f     % 6d "
        "%8.2f %8.2f  % 5.0f "
        "% 5.0f % 5.0f % 5.0f %9.3f "
        "[%5.3f] [%5.3f]\n",
        sh.nchar_, base, " ",
        meanfit, iworst, me17, fm29, iworstfm29,
        InterEyeDist(refshape), EyeMouthDist(refshape), estyaw,
        pose(0), pose(1), pose(2), pose(3),
        facedet_time, asmsearch_time);
}

static void ProcessShapes(
    const ShapeFile& sh,      // in
    FILE*            fitfile) // in
{
    Pacifier pacifier(sh.nshapes_); // only used if quiet_g
    const int nchar = int(floor(log10(double(sh.nshapes_)) + 1)); // align prints
    for (int ishape = 0; ishape < sh.nshapes_; ishape++)
    {
        if (quiet_g)
            pacifier.Print_(ishape);
        else
        {
            if (sh.nshapes_ > 1)
                lprintf("%*d ", nchar, ishape);
            lprintf("%*.*s:%s", sh.nchar_, sh.nchar_,
                    sh.bases_[ishape].c_str(), trace_g? "\n": " ");
        }
        const char* imgpath =
                PathGivenDirs(sh.bases_[ishape].c_str(), sh.dirs_, sh.shapepath_);
        cv::Mat_<unsigned char> img(cv::imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE));
        if (!img.data)
            Err("Cannot load %s", imgpath);

        const clock_t start_time = clock();

        if (!stasm_open_image((const char*)img.data, img.cols, img.rows, imgpath,
                              0 /*multi*/, minwidth_g))
            Err("stasm_open_image failed:  %s", stasm_lasterr());

        clock_t start_time1 = clock();
        const double facedet_time = double(start_time1 - start_time) / CLOCKS_PER_SEC;

        int foundface;
        float estyaw; // estimated yaw
        float landmarks[2 * stasm_NLANDMARKS]; // x,y coords

        if (!stasm_search_auto_ext(&foundface, landmarks, &estyaw))
            Err("stasm_search_auto failed: %s", stasm_lasterr());

        const double search_time = double(clock() - start_time) / CLOCKS_PER_SEC;

        if (ishape == 0)
            logprintf("%s: ", Base(imgpath));

        ProcessFace(img, imgpath, foundface, landmarks, estyaw,
                    facedet_time, search_time, sh.shapes_[ishape], fitfile, sh);

    }
    if (quiet_g)
    {
        pacifier.End_();
        printf("\n");
    }
}

static void main1(int argc, const char** argv)
{
    OpenLogFile();
    print_g = true;
    const bool old_trace = trace_g;
    if (!stasm_init("../data", 1 /*trace*/))
        Err("stasm_init failed %s", stasm_lasterr());
    trace_g = old_trace;

    ShapeFile sh;  // contents of the shape file
    FILE* fitfile; // the fit file we will create

    ProcessOptions(sh, fitfile,
                   argc, argv);

    const clock_t start_time = clock();
    ProcessShapes(sh, fitfile);
    lprintf("[MeanTimePerImg %.3f]\n",
        double(clock() - start_time) / (sh.nshapes_ * CLOCKS_PER_SEC));
    fclose(fitfile);
}

} // namespace stasm

// This application calls Stasm's internal routines.  Thus we need to catch a
// potential throw from Stasm's error handlers.  Hence the try/catch code below.

int main(int argc, const char** argv)
{
    stasm::CatchOpenCvErrs();
    try
    {
        stasm::main1(argc, argv);
    }
    catch(...)
    {
        // a call was made to Err or a CV_Assert failed
        printf("\n%s\n", stasm_lasterr());
        exit(1);
    }
    return 0; // success
}
