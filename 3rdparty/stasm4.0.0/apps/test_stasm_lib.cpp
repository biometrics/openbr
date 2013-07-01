// test_stasm_lib.cpp: test stasm_lib.cpp
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <time.h>
#include "opencv/highgui.h" // needed for imread
#include "stasm_lib.h"
#include "stasm_lib_ext.h"  // needed for stasm_search_auto_ext
#include "stasm_landmarks.h"

#pragma warning(disable:4996) // 'vsprintf': This function may be unsafe

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

static void PrintLandmarks(const float* landmarks, const char* msg)
{
    stasm_printf("%s:\n", msg);
    for (int i = 0; i < stasm_NLANDMARKS; i++)
        stasm_printf("%3d: %4.0f %4.0f\n",
                     i, landmarks[i*2], landmarks[i*2+1]);
}

static void DrawLandmarks(
    cv::Mat_<unsigned char>& img,
    float                    landmarks[],
    int                      nlandmarks = stasm_NLANDMARKS)
{
    for (int i = 0; i < nlandmarks-1; i++)
    {
        const int ix  = cvRound(landmarks[i*2]);       // this point
        const int iy  = cvRound(landmarks[i*2+1]);
        const int ix1 = cvRound(landmarks[(i+1)*2]);   // next point
        const int iy1 = cvRound(landmarks[(i+1)*2+1]);
        cv::line(img,
                 cv::Point(ix, iy), cv::Point(ix1, iy1), 255, 3);
    }
}

int main(int argc, const char** argv)
{
    if (argc != 5)
        Exit("Usage: test_stasm_lib MULTI MINWIDTH TRACE IMAGE");

    const int multi = argv[1][0] - '0';
    if (multi != 0 && multi != 1)
        Exit("Usage: test_stasm_lib MULTI MINWIDTH TRACE IMAGE, "
             "with MULTI 0 or 1, you have MULTI %s", argv[1]);

    int minwidth = -1;
    if (sscanf(argv[2], "%d", &minwidth) != 1 ||
        minwidth < 1 || minwidth > 100)
        {
        Exit("Usage: test_stasm_lib MULTI MINWIDTH TRACE IMAGE with "
             "MINWIDTH 1 to 100,  you have MINWIDTH %s", argv[2]);
        }

    const int trace = argv[3][0] - '0';
    if (trace < 0 || trace > 1)
        Exit("Usage: test_stasm_lib MULTI MINWIDTH TRACE IMAGE, with TRACE 0 or 1");

    if (!stasm_init("../data", trace))
        Exit("stasm_init failed: %s", stasm_lasterr());

    const char* path = argv[4]; // image name
    stasm_printf("Reading %s\n", path);
    const cv::Mat_<unsigned char> img(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE));
    if (!img.data) // could not load image?
        Exit("Cannot load %s", path);

    cv::Mat_<unsigned char> outimg(img.clone());

    if (!stasm_open_image((const char*)img.data, img.cols, img.rows,
                          path, multi != 0, minwidth))
        Exit("stasm_open_image failed: %s", stasm_lasterr());

    // Test stasm_search_auto.
    // The min face size was set in the above stasm_open_image call.

    float landmarks[2 * stasm_NLANDMARKS]; // x,y coords
    int iface = 0;
    while (1)
        {
        stasm_printf("--- Auto Face %d ---\n", iface);
        int foundface;
        float estyaw;
        if (!stasm_search_auto_ext(&foundface, landmarks, &estyaw))
            Exit("stasm_search_auto failed: %s", stasm_lasterr());
        if (!foundface)
            {
            stasm_printf("No more faces\n");
            break; // note break
            }
        char s[100]; sprintf(s, "\nFinal with auto init (estyaw %.0f)", estyaw);
        PrintLandmarks(landmarks, s);
        DrawLandmarks(outimg, landmarks);
        iface++;
        if (trace)
            stasm_printf("\n");
        }
    imwrite("test_stasm_lib_auto.bmp", outimg);
    if (multi == 0 && minwidth == 25 && iface)
        {
        // Test stasm_search_pinned.  A human user is not at hand, so gyp by using
        // points from the last face found above for our 5 start points

        stasm_printf("--- Pinned Face %d ---\n", iface);
        float pinned[2 * stasm_NLANDMARKS]; // x,y coords
        memset(pinned, 0, sizeof(pinned));
        pinned[L_LEyeOuter*2]      = landmarks[L_LEyeOuter*2] + 2;
        pinned[L_LEyeOuter*2+1]    = landmarks[L_LEyeOuter*2+1];
        pinned[L_REyeOuter*2]      = landmarks[L_REyeOuter*2] - 2;
        pinned[L_REyeOuter*2+1]    = landmarks[L_REyeOuter*2+1];
        pinned[L_CNoseTip*2]       = landmarks[L_CNoseTip*2];
        pinned[L_CNoseTip*2+1]     = landmarks[L_CNoseTip*2+1];
        pinned[L_LMouthCorner*2]   = landmarks[L_LMouthCorner*2];
        pinned[L_LMouthCorner*2+1] = landmarks[L_LMouthCorner*2+1];
        pinned[L_RMouthCorner*2]   = landmarks[L_RMouthCorner*2];
        pinned[L_RMouthCorner*2+1] = landmarks[L_RMouthCorner*2+1];

        memset(landmarks, 0, sizeof(landmarks));
        if (!stasm_search_pinned(landmarks,
                pinned, (const char*)img.data, img.cols, img.rows, path))
            Exit("stasm_search_pinned failed: %s", stasm_lasterr());
        PrintLandmarks(landmarks, "Final with pinned init");
        outimg = img.clone();
        DrawLandmarks(outimg, landmarks);
        imwrite("test_stasm_lib_pinned.bmp", outimg);

        // test stasm_convert_shape
        float newlandmarks[2 * stasm_NLANDMARKS]; // x,y coords

        memcpy(newlandmarks, landmarks, 2 * stasm_NLANDMARKS * sizeof(float));
        stasm_convert_shape(newlandmarks, 68);
        PrintLandmarks(newlandmarks, "stasm77 to xm2vts");
#if 0
        outimg = img.clone();
        DrawLandmarks(outimg, newlandmarks, 68);
        imwrite("test_stasm_lib_68.bmp", outimg);
#endif
        memcpy(newlandmarks, landmarks, 2 * stasm_NLANDMARKS * sizeof(float));
        stasm_convert_shape(newlandmarks, 76);
        PrintLandmarks(newlandmarks, "stasm77 to stasm76");
#if 0
        outimg = img.clone();
        DrawLandmarks(outimg, newlandmarks, 76);
        imwrite("test_stasm_lib_76.bmp", outimg);
#endif

#if 0
        memcpy(newlandmarks, landmarks, 2 * stasm_NLANDMARKS * sizeof(float));
        stasm_convert_shape(newlandmarks, 22);
        PrintLandmarks(newlandmarks, "stasm77 to stasm22");
        outimg = img.clone();
        DrawLandmarks(outimg, newlandmarks, 22);
        imwrite("test_stasm_lib_22.bmp", outimg);

        memcpy(newlandmarks, landmarks, 2 * stasm_NLANDMARKS * sizeof(float));
        stasm_convert_shape(newlandmarks, 20);
        PrintLandmarks(newlandmarks, "stasm77 to stasm20");
        outimg = img.clone();
        DrawLandmarks(outimg, newlandmarks, 20);
        imwrite("test_stasm_lib_20.bmp", outimg);
#endif
        }

    return 0;       // success
}
