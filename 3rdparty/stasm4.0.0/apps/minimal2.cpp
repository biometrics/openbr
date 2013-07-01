// minimal2.cpp: Display the landmarks of possibly multiple faces in an image.

#include <stdio.h>
#include <stdlib.h>
#include "opencv/highgui.h"
#include "stasm_lib.h"

static void error(const char* s1, const char* s2)
{
    printf("Stasm version %s: %s %s\n", stasm_VERSION, s1, s2);
    exit(1);
}

int main()
{
    if (!stasm_init("../data", 0 /*trace*/))
        error("stasm_init failed: ", stasm_lasterr());

    static const char* path = "../data/testface.jpg";

    cv::Mat_<unsigned char> img(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE));

    if (!img.data)
        error("Cannot load", path);

    if (!stasm_open_image((const char*)img.data, img.cols, img.rows, path,
                          1 /*multiface*/, 10 /*minwidth*/))
        error("stasm_open_image failed: ", stasm_lasterr());

    int foundface;
    float landmarks[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)

    int nfaces = 0;
    while (1)
    {
        if (!stasm_search_auto(&foundface, landmarks))
             error("stasm_search_auto failed: ", stasm_lasterr());

        if (!foundface)
            break;      // note break

        // for demonstration, convert from Stasm 77 points to XM2VTS 68 points
        stasm_convert_shape(landmarks, 68);

        // draw the landmarks on the image as white dots
        stasm_force_points_into_image(landmarks, img.cols, img.rows);
        for (int i = 0; i < stasm_NLANDMARKS; i++)
            img(cvRound(landmarks[i*2+1]), cvRound(landmarks[i*2])) = 255;

        nfaces++;
    }
    printf("%s: %d face(s)\n", path, nfaces);
    fflush(stdout);
    cv::imwrite("minimal2.bmp", img);
    cv::imshow("stasm minimal2", img);
    cv::waitKey();

    return 0;
}
