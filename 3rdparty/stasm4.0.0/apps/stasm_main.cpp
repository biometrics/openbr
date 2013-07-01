// stasm_main.cpp: command line utility to run Stasm on a set of images
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"
#include "opencv/highgui.h"
#include "appmisc.h"

namespace stasm
{
static bool writeimgs_g = true; // -i flag
static bool multiface_g;        // -m flag
static int  minwidth_g = 25;    // -s flag
static int  nlandmarks_g = 77;  // -n flag
static bool csv_g;              // -c flag

//-----------------------------------------------------------------------------

static void GetOptions(
    int&          argc, // in
    const char**& argv) // io
{
    static const char* usage =
        "stasm [FLAGS] IMAGE [IMAGE2 ...]\n"
        "\n"
        "Landmark results are saved in stasm.log\n"
        "\n"
        "Example: stasm face.jpg\n"
        "\n"
        "Flags:\n"
        "    -m     multiface\n"
        "    -s     small faces (min width 10%% of image width, the default is 25%%)\n"
        "    -n N   save as 77 (default), 76, 68, 22, 20, or 17 point shape\n"
        "    -c     save landmarks as CSVs (default saves in shapefile format)\n"
        "    -i     do not write landmarked images (faster)\n"
        "    -d     enable debug prints\n"
        "    -?     help\n"
        "\n"
        "Stasm version %s    www.milbo.users.sonic.net/stasm\n"; // %s is stasm_VERSION

    if (argc < 2)
        Err("No image.  Use stasm -? for help.");
    while (argc-- > 0 && (*++argv)[0] == '-')
    {
        if ((*argv + 1)[1])
            Err("Invalid flag -%s (you cannot combine flags).  Use stasm -? for help.",
                *argv + 1);
        switch ((*argv + 1)[0])
        {
            case 'c':
                csv_g = true;
                break;
            case 'd':
                trace_g = true; // trace_g defined in print.cpp
                break;
            case 'i':
                writeimgs_g = false;
                break;
            case 'm':
                multiface_g = true;
                break;
            case 's':
                minwidth_g = 10;
                break;
            case 'n':
                if (argc < 3)
                    Err("-n flag must be followed by NLANDMARKS.  For example -n 68");
                argc--;
                argv++;
                nlandmarks_g = -1;
                if (1 != sscanf(*argv, "%d", &nlandmarks_g) || nlandmarks_g < 1)
                    Err("-n flag must be followed by NLANDMARKS.  For example -n 68");
                // validity of nlandmarks_g will be checked later after call to ConvertShape
                break;
            case '?':
                printf(usage, stasm_VERSION);
                exit(1);
            default:
                Err("Invalid flag -%s.  Use stasm -? for help.", *argv + 1);
                break;
        }
    }
    if (argc < 1)
        Err("No image.  Use stasm -? for help.");
}

static void ProcessFace(
    CImage&      cimg,      // io: color version of image
    const float* landmarks, // in
    int          nfaces,    // in
    const char*  imgpath)   // in
{
    Shape shape(LandmarksAsShape(landmarks));
    shape = ConvertShape(shape, nlandmarks_g);
    if (shape.rows == 0)
        Err("Cannot convert to a %d point shape", nlandmarks_g);
    if (writeimgs_g)
        DrawShape(cimg, shape);
    if (multiface_g)
    {
        logprintf("face %d\n", nfaces);
        double xmin, xmax, ymin, ymax; // needed to position face nbr
        ShapeMinMax(xmin, xmax, ymin, ymax, shape);
        ImgPrintf(cimg,
                  (xmin + xmax)/2, ymin - (ymax-ymin)/50., 0xff0000, 1,
                 "%d", nfaces);
    }
    if (csv_g)
        LogShapeAsCsv(shape, imgpath);
    else
        LogShape(shape, imgpath);
}

static void ProcessImg(
    const char* imgpath) // in
{
    Image img(cv::imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE));
    if (!img.data)
        Err("Cannot load %s", imgpath);
    if (!stasm_open_image((const char*)img.data, img.cols, img.rows, imgpath,
                          multiface_g, minwidth_g))
        Err("stasm_open_image failed:  %s", stasm_lasterr());

    CImage cimg;     // color version of image
    if (writeimgs_g) // actually need the color image?
        cvtColor(img, cimg, CV_GRAY2BGR);
    int nfaces = 0;
    while (1)
    {
        if (trace_g && nfaces > 0 && multiface_g)
            stasm_printf("\n%d: ", nfaces);

        int foundface;
        float landmarks[2 * stasm_NLANDMARKS]; // x,y coords
        if (!stasm_search_auto(&foundface, landmarks))
            Err("stasm_search_auto failed: %s", stasm_lasterr());

        if (!foundface)
            break; // note break

        ProcessFace(cimg, landmarks, nfaces, Base(imgpath));
        nfaces++;
    }
    if (trace_g)
        lprintf("\n");
    if (writeimgs_g && nfaces)
    {
        // write as a bmp not as a jpg because don't want blurred shape lines
        char newpath[SLEN]; sprintf(newpath, "%s_stasm.bmp", Base(imgpath));
        lprintf("%s ", newpath);
        if (!cv::imwrite(newpath, cimg))
            Err("Could not write %s", newpath);
    }
    lprintf("%d face%s\n", nfaces, plural(nfaces));
}

static void main1(int argc, const char** argv)
{
    GetOptions(argc, argv);
    OpenLogFile();

    // argc is now the number of images and argv is the image filenames

    const bool old_trace = trace_g;
    if (!stasm_init("../data", 1 /*trace*/))
        Err("stasm_init failed %s", stasm_lasterr());
    trace_g = old_trace;

    const int ndigits = int(floor(log10(double(argc)) + 1)); // for aligning

    for (int i_img = 0; i_img < argc; i_img++)
    {
        const char* const imgpath = argv[i_img];
        if (argc > 1)
            lprintf("%*d ", ndigits, i_img);
        lprintf("%s: ", imgpath);
        ProcessImg(imgpath);
    }
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
    return 0;       // success
}
