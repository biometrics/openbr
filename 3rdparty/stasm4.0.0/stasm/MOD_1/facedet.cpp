// facedet.cpp: find faces in images (frontal model version)
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "../stasm.h"

namespace stasm
{
typedef vector<DetPar> vec_DetPar;

//static cv::CascadeClassifier facedet_g;  // the face detector

static double BORDER_FRAC = 0.1; // fraction of image width or height
                                 // use 0.0 for no border

//-----------------------------------------------------------------------------

void FaceDet::OpenFaceDetector_( // called by stasm_init, init face det from XML file
    const char* datadir,         // in: directory of face detector files
    void*)                       // in: unused (func signature compatibility)
{
    (void) datadir;
    //OpenDetector(facedet_g, "haarcascade_frontalface_alt2.xml",  datadir);
}

// If a face is near the edge of the image, the OpenCV detectors tend to
// return a too-small face rectangle.  By adding a border around the edge
// of the image we mitigate this problem.

static Image EnborderImg(    // return the image with a border
    int&         leftborder, // out: border size in pixels
    int&         topborder,  // out: border size in pixels
    const Image& img)        // io
{
    Image bordered_img(img);
    leftborder = cvRound(BORDER_FRAC * bordered_img.cols);
    topborder  = cvRound(BORDER_FRAC * bordered_img.rows);
    copyMakeBorder(bordered_img, bordered_img,
                   topborder, topborder, leftborder, leftborder,
                   cv::BORDER_REPLICATE);
    return bordered_img;
}

void DetectFaces(          // all face rects into detpars
    vec_DetPar&  detpars,  // out
    const Image& img,      // in
    int          minwidth,
    cv::CascadeClassifier cascade) // in: as percent of img width
{
    int leftborder = 0, topborder = 0; // border size in pixels
    Image bordered_img(BORDER_FRAC == 0?
                       img: EnborderImg(leftborder, topborder, img));

    // Detection results are very slightly better with equalization
    // (tested on the MUCT images, which are not pre-equalized), and
    // it's quick enough to equalize (roughly 10ms on a 1.6 GHz laptop).

    Image equalized_img; cv::equalizeHist(bordered_img, equalized_img);

    CV_Assert(minwidth >= 1 && minwidth <= 100);

    int minpix = MAX(100, cvRound(img.cols * minwidth / 100.));

    // the params below are accurate but slow
    static const double SCALE_FACTOR   = 1.1;
    static const int    MIN_NEIGHBORS  = 3;
    static const int    DETECTOR_FLAGS = 0;

    vec_Rect facerects = // all face rects in image
        Detect(equalized_img, &cascade, NULL,
               SCALE_FACTOR, MIN_NEIGHBORS, DETECTOR_FLAGS, minpix);

    // copy face rects into the detpars vector

    detpars.resize(NSIZE(facerects));
    for (int i = 0; i < NSIZE(facerects); i++)
    {
        Rect* facerect = &facerects[i];
        DetPar detpar; // detpar constructor sets all fields INVALID
        // detpar.x and detpar.y is the center of the face rectangle
        detpar.x = facerect->x + facerect->width / 2.;
        detpar.y = facerect->y + facerect->height / 2.;
        detpar.x -= leftborder; // discount the border we added earlier
        detpar.y -= topborder;
        detpar.width  = double(facerect->width);
        detpar.height = double(facerect->height);
        detpar.yaw = 0; // assume face has no yaw in this version of Stasm
        detpar.eyaw = EYAW00;
        detpars[i] = detpar;
    }
}

// order by increasing distance from left marg, and dist from top marg within that

static bool IncreasingLeftMargin( // compare predicate for std::sort
    const DetPar& detpar1,        // in
    const DetPar& detpar2)        // in
{
    return 1e5 * detpar2.x + detpar2.y >
           1e5 * detpar1.x + detpar1.y;
}

// order by decreasing width, and dist from the left margin within that

static bool DecreasingWidth( // compare predicate for std::sort
    const DetPar& detpar1,   // in
    const DetPar& detpar2)   // in
{
    return 1e5 * detpar2.width - detpar2.x <
           1e5 * detpar1.width - detpar1.x;

}

// Discard too big or small faces (this helps reduce the number of false positives)

static void DiscardMissizedFaces(
    vec_DetPar& detpars) // io
{
    // constants (TODO These have not yet been rigorously empirically adjusted.)
    const double MIN_WIDTH = 1.33; // as fraction of median width
    const double MAX_WIDTH = 1.33; // as fraction of median width

    if (NSIZE(detpars) >= 3) // need at least 3 faces
    {
        // sort the faces on their width (smallest first) so can get median width
        sort(detpars.begin(), detpars.end(), DecreasingWidth);
        const int median     = cvRound(detpars[NSIZE(detpars) / 2].width);
        const int minallowed = cvRound(median / MIN_WIDTH);
        const int maxallowed = cvRound(MAX_WIDTH * median);
        // keep only faces that are not too big or small
        vec_DetPar all_detpars(detpars);
        detpars.resize(0);
        for (int iface = 0; iface < NSIZE(all_detpars); iface++)
        {
            DetPar* face = &all_detpars[iface];
            if (face->width >= minallowed && face->width <= maxallowed)
                detpars.push_back(*face);
            else if (trace_g || TRACE_IMAGES)
                lprintf("[discard %d of %d]", iface, NSIZE(all_detpars));
        }
    }
}

#if TRACE_IMAGES // will be 0 unless debugging (defined in stasm.h)

static void TraceFaces(        // write image showing detected face rects
    const vec_DetPar& detpars, // in
    const Image&      img,     // in
    const char*       path)    // in
{
    (void) detpars;
    (void) img;
    (void) filename;

    CImage cimg; cvtColor(img, cimg, CV_GRAY2BGR); // color image
    for (int iface = 0; iface < NSIZE(detpars); iface++)
    {
        const DetPar &detpar = detpars[iface];

        rectangle(cimg,
                  cv::Point(cvRound(detpar.x - detpar.width/2),
                            cvRound(detpar.y - detpar.height/2)),
                  cv::Point(cvRound(detpar.x + detpar.width/2),
                            cvRound(detpar.y + detpar.height/2)),
                  CV_RGB(255,255,0), 2);

        ImgPrintf(cimg, // 10 * iface to minimize overplotting
                  detpar.x + 10 * iface, detpar.y, 0xffff00, 1, ssprintf("%d", iface));
    }

    lprintf("%s\n", path);
    if (!cv::imwrite(path, cimg))
        Err("Cannot write %s", path);
}

#endif

void FaceDet::DetectFaces_(  // call once per image to find all the faces
    const Image& img,        // in: the image (grayscale)
    const char*,             // in: unused (match virt func signature)
    bool         multiface,  // in: if false, want only the best face
    int          minwidth,   // in: min face width as percentage of img width
    void*        user,       // in: unused (match virt func signature)
    cv::CascadeClassifier cascade)
{
    CV_Assert(user == NULL);
    //CV_Assert(!facedet_g.empty()); // check that OpenFaceDetector_ was called

    DetectFaces(detpars_, img, minwidth, cascade);
    DiscardMissizedFaces(detpars_);
    if (multiface) // order faces on increasing distance from left margin
    {
        sort(detpars_.begin(), detpars_.end(), IncreasingLeftMargin);
    }
    else
    {
        // order faces on decreasing width, keep only the first (the largest face)
        sort(detpars_.begin(), detpars_.end(), DecreasingWidth);
        if (NSIZE(detpars_))
            detpars_.resize(1);
    }
    iface_ = 0; // next invocation of NextFace_ must get first face
}

// Get the (next) face from the image.
// If no face available, return detpar.x INVALID.
// Eyes, mouth, and rot in detpar always returned INVALID.

const DetPar FaceDet::NextFace_(void)
{
    DetPar detpar; // detpar constructor sets all fields INVALID

    if (iface_ < NSIZE(detpars_))
        detpar = detpars_[iface_++];

    return detpar;
}

} // namespace stasm
