// shapetostasm31.cpp: Convert a shape file to a Stasm version 3.1 shape file
//
// This will automatically convert 77 point shapes to 76 point shapes.
//
// This converts:
//   o the point location/numbering if a 77 point shape
//   o the tags
//   o the coords (OpenCV to old style Stasm coords).
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

#pragma warning(disable:4996) // 'sprintf': This function may be unsafe

namespace stasm
{
static unsigned NewBits(unsigned bits) // convert tag bits to Stasm 3.1 format
{
    // not implemented: FA_NnFailed, FA_Synthesize, FA_VjFailed, FA_Rowley

    unsigned newbits = 0; // stasm 3.1 tag bits

    newbits |= (AT_BadImg & bits)?     FA_BadImage:   0;
    newbits |= (AT_Glasses & bits)?    FA_Glasses:    0;
    newbits |= (AT_Beard & bits)?      FA_Beard:      0;
    newbits |= (AT_Mustache & bits)?   FA_Mustache:   0;
    newbits |= (AT_Obscured & bits)?   FA_Obscured:   0;
    newbits |= (AT_BadEye & bits)?     FA_EyesClosed: 0;
    newbits |= (AT_Expression & bits)? FA_Expression: 0;
    newbits |= (AT_EYAW00 & bits)?     FA_ViolaJones: 0;

    return newbits;
}

static void Print(FILE* file, double x, const char* msg)
{
    Fprintf(file, int(x) == x? "%.0f%s":  "%.1f%s", x, msg);
}

static void main1(int argc, const char** argv)
{
    print_g = true;
    if (argc != 2)
        Err("Usage: shapetostasm31 file.shape");
    ShapeFile sh; // contents of the shape file
    sh.Open_(argv[1]);
    if (sh.shapes_[0].rows == 77)
        lprintf("Converting 77 point to 76 point shapes\n");
    char newpath[SLEN];
    sprintf(newpath, "%s_stasm31.shape", Base(argv[1]));
    lprintf("Creating %s ", newpath);
    FILE* file = fopen(newpath, "w");
    if (!file)
        Err("Cannot open %s for writing", newpath);
    Fprintf(file, "ss %s\n\n", newpath);
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

        Shape shape;
        if (sh.shapes_[0].rows == 77)
            shape = (ConvertShape(sh.shapes_[ishape], 76)); // convert 76 point shape
        else
            shape = sh.shapes_[ishape].clone();

        Fprintf(file, "\"%4.4x %s\"\n",
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
                double oldx = shape(i, IX) - cols2;
                double oldy = rows2 - shape(i, IY) - 1;
                if (!PointUsed(oldx, oldy))
                    oldx = XJITTER;
                Print(file, oldx, " ");
                Print(file, oldy, "\n");
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
