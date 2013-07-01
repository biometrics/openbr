// hat_example.cpp: Load a face and print the HAT descriptor for a few points.
//                  This is an example of how to use HAT descriptors in your
//                  own code, without Stasm.

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#if _MSC_VER // microsoft compiler
#pragma warning(disable:4996) // disable warning in cv: sprintf: This function may be unsafe
#endif
#include "opencv/cv.h"
#include "opencv/highgui.h" // needed only to read the file off disk

// a few defines needed by hat.h since we aren't including stasm.h
typedef cv::Mat_<double> MAT;   // a matrix with double elements
typedef cv::Mat_<double> VEC;   // by convention indicates one-dim matrix
typedef cv::Mat_<byte> Image;   // a gray image (a matrix of bytes)
typedef std::vector<int>    vec_int;
typedef std::vector<double> vec_double;
#define DISALLOW_COPY_AND_ASSIGN(ClassName) \
    ClassName(const ClassName&); void operator=(const ClassName&)

#include "hat.h"

int main()
{
    static const char* path = "../data/testface.jpg";
    Image img(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE));
    if (!img.data)
    {
        printf("Cannot load %s\n", path);
        exit(1);
    }

    // Hat::Init_ must be called once per image (and thereafter you
    // can call Hat::Desc_ as many times as you need on the same image).
    // The patch width (9 in this example) must be odd (else you will
    // get a CV_Assert fail in hat.cpp).

    stasm::Hat hat; hat.Init_(img, 9);

    // Get the HAT the descriptor at image coords 12,34.

    cv::Mat_<double> desc(hat.Desc_(12, 34));

    // Print the descriptor elements.

    printf("%d element HAT desc at 12,34 is", desc.total());
    for (int i = 0; i < int(desc.total()); i++)
        printf(" %g", desc(i));
    printf("\n");

    return 0;
}
