// shapefromstasm31.cpp: convert a Stasm 3.1 shapefile to the current Stasm version
//
// This converts the tags and the coords (old style Stasm coords to OpenCV).
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include <sys/types.h>
#include <sys/stat.h>
#include "stasm.h"
#include "opencv/highgui.h"
#include "atface.h"
#include "stasm_regex.h"
#include "shapefile.h"
#include "appmisc.h"

#if _MSC_VER // microsoft compiler
#pragma warning(disable:4996) // 'sprintf': This function may be unsafe
#endif

namespace stasm
{
static unsigned NewBits(unsigned bits) // convert tag bits from Stasm 3.1 format
{
    unsigned NewBits = 0;

    NewBits |= (FA_BadImage & bits)?   AT_BadImg:     0;
    NewBits |= (FA_Glasses & bits)?    AT_Glasses:    0;
    NewBits |= (FA_Beard & bits)?      AT_Beard:      0;
    NewBits |= (FA_Mustache & bits)?   AT_Mustache:   0;
    NewBits |= (FA_Obscured & bits)?   AT_Obscured:   0;
    NewBits |= (FA_EyesClosed & bits)? AT_BadEye:     0;
    NewBits |= (FA_Expression & bits)? AT_Expression: 0;
//  NewBits |= (FA_NnFailed & bits)?
//  NewBits |= (FA_Synthesize & bits)?
//  NewBits |= (FA_VjFailed & bits)?
    NewBits |= (FA_ViolaJones & bits)? AT_EYAW00:     0;
//  NewBits |= (FA_Rowley & bits)?

    return NewBits;
}

static void Print(FILE* file, double x, const char* msg)
{
    Fprintf(file, int(x) == x? "%.0f%s":  "%.1f%s", x, msg);
}

static void main1(int argc, const char** argv)
{
    print_g = true;
    if (argc != 2)
        Err("Usage: shapefromstasm31 file.shape");
    ShapeFile sh; // contents of the shape file
    sh.Open_(argv[1]);
    char newpath[SLEN];
    sprintf(newpath, "%s_stasm35.shape", Base(argv[1]));
    lprintf("Creating %s ", newpath);
    FILE* file = fopen(newpath, "w");
    if (!file)
        Err("Cannot open %s for writing", newpath);
    Fprintf(file, "shape %s\n\n", newpath);
    Fprintf(file, "Directories %s\n\n", sh.dirs_);
    Pacifier pacifier(sh.nshapes_);
    for (int ishape = 0; ishape < sh.nshapes_; ishape++)
    {
        // we need the image width and height to convert the coords
        const char* imgpath =
                PathGivenDirs(sh.bases_[ishape].c_str(), sh.dirs_, sh.shapepath_);
        cv::Mat_<unsigned char> img(cv::imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE));
        if (!img.data)
            Err("Cannot load %s", imgpath);

        Shape shape(sh.shapes_[ishape].clone());

        Fprintf(file, "%8.8x %s\n",
            NewBits(sh.bits_[ishape]), sh.bases_[ishape].c_str());
        Fprintf(file, "{ %d %d\n", shape.rows, shape.cols);

        const int cols2 = img.cols / 2;
        const int rows2 = img.rows / 2;

        for (int i = 0; i < shape.rows; i++)
        {
            if (!PointUsed(shape, i))
                Fprintf(file, "0 0\n");
            else
            {
                double newx = shape(i, IX) + cols2;
                double newy = rows2 - shape(i, IY) - 1;
                if (!PointUsed(newx, newy))
                    newx = XJITTER;
                Print(file, newx, " ");
                Print(file, newy, "\n");
            }
        }
        Fprintf(file, "}\n");
        pacifier.Print_(ishape);
    }
    pacifier.End_();
    fclose(file);
    lprintf("\n");
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
