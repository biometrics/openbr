// test_stasm_lib_err.cpp: test error handling in stasm_lib.cpp
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <time.h>
#include "opencv/highgui.h" // needed for imread
#include "stasm_lib.h"

#pragma warning(disable:4996) // 'vsprintf': This function  may be unsafe

static void Exit(const char* format, ...) // args like printf
{
    char s[1024+1];
    va_list args;
    va_start(args, format);
    vsprintf(s, format, args);
    va_end(args);
    stasm_printf("\n%s\n", s);
    exit(1);
}

int main(void)
{
    static const char* path = "../data/testface.jpg";
    const cv::Mat_<unsigned char> img(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE));
    if (!img.data) // could not load image?
        Exit("Cannot load %s", path);

    if (!stasm_init("../data", 1 /*trace*/))
        Exit("stasm_init failed: %s", stasm_lasterr());

    // Pass a NULL path to stasm_open_image, to force an error.
    // The intention is just to test the overall error handling structure.
    int opened = stasm_open_image((const char*)img.data, img.cols, img.rows,
                                  NULL, 0, 25);
    if (opened == 0)
    {
        stasm_printf("Expect assertion fail here: ");
        stasm_printf("%s\n", stasm_lasterr());
    }
    else // should not get here
        Exit("stasm_open_image did not fail as expected");

    return 0;
}
