// shapeconvert.cpp: Convert a shapefile to a different number of points.

#include <sys/types.h>
#include <sys/stat.h>
#include "stasm.h"
#include "shapefile.h"
#include "appmisc.h"

#pragma warning(disable:4996) // 'sprintf': This function may be unsafe

static int nlandmarks_g = 77;  // -n flag

namespace stasm
{
static void GetOptions(
    int&          argc, // in
    const char**& argv) // io
{
    static const char* usage =
        "shapeconvert [FLAGS] file.shape\n"
        "\n"
        "Flags:\n"
        "    -n NPOINTS   convert to NPOINTS\n"
        "    -?           help\n";
    if (argc < 2)
        Err("No shapefile.  Use shapeconvert -? for help.");
    while (argc-- > 0 && (*++argv)[0] == '-')
    {
        if ((*argv + 1)[1])
            Err("Invalid flag -%s (you cannot combine flags).  Use shapeconvert -? for help.",
                *argv + 1);
        switch ((*argv + 1)[0])
        {
            case 'n':
                if (argc < 3)
                    Err("-n flag must be followed by NPOINTS.  For example -n 17");
                argc--;
                argv++;
                nlandmarks_g = -1;
                if (1 != sscanf(*argv, "%d", &nlandmarks_g) || nlandmarks_g < 1)
                    Err("-n flag must be followed by NPOINTS.  For example -n 17");
                // validity of nlandmarks_g will be checked later after call to ConvertShape
                break;
            case '?':
                printf(usage);
                exit(1);
            default:
                Err("Invalid flag -%s.  Use shapeconvert -? for help.", *argv + 1);
                break;
        }
    }
    if (argc != 1)
        Err("No shapefile.  Use shapeconvert -? for help.");
}

static void Print(FILE* file, double x, const char* msg)
{
    Fprintf(file, int(x) == x? "%.0f%s":  "%.1f%s", x, msg);
}

static void main1(int argc, const char** argv)
{
    GetOptions(argc, argv);
    OpenLogFile();
    print_g = true;
    ShapeFile sh;      // contents of the shape file
    sh.Open_(argv[0]); // argv[0] is the shapefile path
    char newpath[SLEN];
    sprintf(newpath, "%s_%d.shape", Base(argv[0]), nlandmarks_g);
    lprintf("Creating %s ", newpath);
    FILE* file = fopen(newpath, "w");
    if (!file)
        Err("Cannot open %s for writing", newpath);
    Fprintf(file, "shape %s\n\n", newpath);
    Fprintf(file, "Directories %s\n\n", sh.dirs_);
    Pacifier pacifier(sh.nshapes_);
    for (int ishape = 0; ishape < sh.nshapes_; ishape++)
    {
        Shape shape(sh.shapes_[ishape]);
        Shape newshape(nlandmarks_g? ConvertShape(shape, nlandmarks_g): shape);
        Fprintf(file, "\"%8.8x %s\"\n", sh.bits_[ishape], sh.bases_[ishape].c_str());
        Fprintf(file, "{ %d %d\n", newshape.rows, newshape.cols);
        for (int i = 0; i < newshape.rows; i++)
        {
           Print(file, newshape(i, IX), " ");
           Print(file, newshape(i, IY), "\n");
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
