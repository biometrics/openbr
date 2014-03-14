// asm.cpp: Active Shape Model class
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"
#include "stasmhash.h"

namespace stasm
{
static void TraceShape(  // write an image file showing current shape on the image
    const Shape& shape,  // in: current search shape
    const Image& pyrimg, // in: image scaled to this pyramid level
    int          ilev,   // in: pyramid level (0 is full size)
    int          iter,   // in: model iteration (-1 if start shape)
    const char*  name)   // in
{
    (void) shape;
    (void) pyrimg;
    (void) ilev;
    (void) iter;
    (void) name;
#if TRACE_IMAGES // will be 0 unless debugging (defined in stasm.h)

    static int index = 0; // number images so they appear in order in the directory

    if (index == 0)
        lprintf("TRACE_IMAGES ");

    Image img; // pyrimg rescaled back to full size image (after rescaling by eyemouth)
    const double RESOLUTION = 2; // 1 for no extra resolution, 2 for double resolution
    const double rescale = RESOLUTION / GetPyrScale(ilev);
    cv::resize(pyrimg, img, cv::Size(), rescale, rescale, cv::INTER_NEAREST);
    CImage cimg; cvtColor(img, cimg, CV_GRAY2BGR); // color image
    DesaturateImg(cimg);
    Shape shape1(shape.clone()); RoundMat(shape1);
    shape1 += .4; // put shape points in center of rescaled pixels
    shape1 *= rescale;
    DrawShape(cimg, shape1, 0xffff00, false, 1);
    char path[SLEN];
    if (iter < 0) // start shape?
        sprintf(path, "_%2.2d_%s.bmp", index, name);
    else
        sprintf(path, "_%2.2d_lev%d_iter%d_%s.bmp", index, ilev, iter, name);
    ImgPrintf(cimg, 10 * RESOLUTION, 20 * RESOLUTION, 0xffff00, 2, path);
    if (iter >= 0)
    {
        // draw 1D patch boundary at one point (patch is drawn
        // horizontal, not rotated to shape boundary as it should be)

        Shape shape2(shape); RoundMat(shape2);
        int ipoint = shape2.rows >= L_RJaw11? L_RJaw11: 0;
        int proflen = 9;
        int x1 = cvRound(shape2(ipoint, IX)) - proflen / 2;
        int x2 = x1 + proflen;
        int y1 = cvRound(shape2(ipoint, IY));
        int y2 = y1 + 1;
        rectangle(cimg,
                  cv::Point(cvRound(rescale * x1), cvRound(rescale * y1)),
                  cv::Point(cvRound(rescale * x2), cvRound(rescale * y2)),
                  CV_RGB(255,0,0), 1);

        // draw 2D patch boundary at one point

        if (ilev <= HAT_START_LEV) // we use HATs only at upper pyr levs
        {
            ipoint = shape2.rows >= L_LPupil? L_LPupil: 0;
            #define round2(x) 2 * cvRound((x) / 2)
            int patchwidth = HAT_PATCH_WIDTH + round2(ilev * HAT_PATCH_WIDTH_ADJ);
            x1 = cvRound(shape2(ipoint, IX)) - patchwidth / 2;
            x2 = x1 + patchwidth;
            y1 = cvRound(shape2(ipoint, IY)) - patchwidth / 2;
            y2 = y1 + patchwidth;
            rectangle(cimg,
                      cv::Point(cvRound(rescale * x1), cvRound(rescale * y1)),
                      cv::Point(cvRound(rescale * x2), cvRound(rescale * y2)),
                      CV_RGB(255,0,0), 1);

        }
    }

    cv::imwrite(path, cimg);
    index++;

#endif // TRACE_IMAGES
}

#if _OPENMP

void Mod::SuggestShape_( // args same as non OpenMP version, see below
    Shape&       shape,  // io
    int          ilev,   // in
    const Image& img,    // in
    const Shape& pinned) // in
const
{
    static bool firsttime = true;
    int ncatch = 0;
    const Shape inshape(shape.clone());

    // Call the search function DescSearch_ concurrently for multiple points.
    // Note that dynamic OpenMP scheduling is faster here than static,
    // because the time through the loop varies widely (mainly because
    // classic descriptors are faster than HATs).

    #pragma omp parallel for schedule(dynamic)

    for (int ipoint = 0; ipoint < shape.rows; ipoint++)
        if (pinned.rows == 0 || !PointUsed(pinned, ipoint)) // skip point if pinned
        {
            // You are not allowed to jump out of an OpenMP for loop.  Thus
            // we need this try block, to subsume the global try blocks in
            // stasm_lib.cpp.  Without this try, a call to Err would cause
            // a jump to the global catch.

            try
            {
                if (firsttime && omp_get_thread_num() == 0)
                {
                    firsttime = false;
                    logprintf("[nthreads %d]", omp_get_num_threads());
                }
                descmods_[ilev][ipoint]->DescSearch_(shape(ipoint, IX), shape(ipoint, IY),img, inshape, ilev, ipoint);
            }
            catch(...)
            {
                ncatch++; // a call was made to Err or a CV_Assert failed
            }
        }

    if (ncatch)
    {
        if (ncatch > 1)
            lprintf("\nMultiple errors, only the first will be printed\n");
        // does not matter what we throw, will be caught by global catch
        throw "SuggestShape_";
    }
}

#else // not _OPENMP

void Mod::SuggestShape_( // estimate shape by matching descr at each point
    Shape&       shape,  // io: points will be moved for best descriptor matches
    int          ilev,   // in: pyramid level (0 is full size)
    const Image& img,    // in: image scaled to this pyramid level
    const Shape& pinned, // in: if no rows then no pinned landmarks, else
                         //     points except those equal to 0,0 are pinned
    const Hat &hat,
    StasmHash &hash)
const
{
    const Shape inshape(shape.clone());

    for (int ipoint = 0; ipoint < shape.rows; ipoint++)
        if (pinned.rows == 0 || !PointUsed(pinned, ipoint)) // skip point if pinned
        {
            // Call the ClassicDescMod or HatDescMod search function
            // to update the current point in shape (the point at ipoint)
            // yaw00.h:YAW00_DESCMODS defines which descriptor model is used
            // for each point (assuming we are using the yaw00 model).

            descmods_[ilev][ipoint]->DescSearch_(shape(ipoint, IX), shape(ipoint, IY),img, inshape, ilev, ipoint, hat, hash);
        }
}
#endif // not _OPENMP

void Mod::LevSearch_(         // do an ASM search at one level in the image pyr
    Shape&       shape,       // io: the face shape for this pyramid level
    int          ilev,        // in: pyramid level (0 is full size)
    const Image& img,         // in: image scaled to this pyramid level
    const Shape& pinnedshape) // in: if no rows then no pinned landmarks, else
                              //     points except those equal to 0,0 are pinned
const
{
    StasmHash hash;

    TraceShape(shape, img, ilev, 0, "enterlevsearch");

    const Hat &hat = InitHatLevData(img, ilev); // init internal HAT mats for this lev

    VEC b(NSIZE(shapemod_.eigvals_), 1, 0.); // eigvec weights, init to 0

    for (int iter = 0; iter < SHAPEMODEL_ITERS; iter++)
    {
        // suggest shape by descriptor matching at each landmark

        SuggestShape_(shape, ilev, img, pinnedshape, hat, hash);

        TraceShape(shape, img, ilev, iter, "suggested");

        // adjust suggested shape to conform to the shape model

        if (pinnedshape.rows)
            shape = shapemod_.ConformShapeToMod_Pinned_(b, shape, ilev, pinnedshape);
        else
            shape = shapemod_.ConformShapeToMod_(b, shape, ilev);

        TraceShape(shape, img, ilev, iter, "conformed");
    }
}

static void CreatePyr(    // create image pyramid
    vector<Image>& pyr,   // out: the pyramid, pyr[0] is full size image
    const Image&   img,   // in:  full size image
    int            nlevs) // in
{
    CV_Assert(nlevs >= 1 && nlevs < 10); // 10 is arb
    pyr.resize(nlevs);
    pyr[0] = img;        // pyramid level 0 is full size image
    for (int ilev = 1; ilev < nlevs; ilev++)
    {
        const double scale = GetPyrScale(ilev);
        cv::resize(img, pyr[ilev], cv::Size(), scale, scale, cv::INTER_LINEAR);
    }
}

Shape Mod::ModSearch_(            // returns coords of the facial landmarks
        const Shape& startshape,  // in: startshape roughly positioned on face
        const Image& img,         // in: grayscale image (typically just ROI)
        const Shape* pinnedshape) // in: pinned landmarks, NULL if nothing pinned
const
{
    Image scaledimg;         // image scaled to fixed eye-mouth distance
    const double imgscale = EYEMOUTH_DIST / EyeMouthDist(startshape);

    // TODO This resize is quite slow (cv::INTER_NEAREST is even slower, why?).
    cv::resize(img, scaledimg,
               cv::Size(), imgscale, imgscale, cv::INTER_LINEAR);

    TraceShape(startshape * imgscale, scaledimg, 0, -1, "start");

    vector<Image> pyr;       // image pyramid (a vec of images, one for each pyr lev)
    CreatePyr(pyr, scaledimg, N_PYR_LEVS);

    Shape shape(startshape * imgscale * GetPyrScale(N_PYR_LEVS));

    Shape pinned;            // pinnedshape scaled to current pyr lev
    if (pinnedshape)
        pinned = *pinnedshape * imgscale * GetPyrScale(N_PYR_LEVS);

    for (int ilev = N_PYR_LEVS-1; ilev >= 0; ilev--)
    {
        shape  *= PYR_RATIO; // scale shape to this pyr lev
        pinned *= PYR_RATIO;

        LevSearch_(shape, ilev, pyr[ilev], pinned);
    }
    return shape / imgscale;
}

} // namespace stasm
