// startshape.cpp: routines for finding the start shape for an ASM search
//
// The model "estart" determines the method we use to create the start shape.
// (The InitMods function initializes estart during Stasm initialization.)
// The current open-source version of Stasm uses estart=ESTART_EYES.
//
// 1. With the model estart=ESTART_RECT_ONLY, the start shape is created by
// aligning the model mean face shape to the face rectangle.  (The face
// rectangle is found by the face detector prior to calling routines in
// this file.)
//
// 2. With the model estart=ESTART_EYES (currently used for the frontal
// model), the start shape is created as follows.  Using the face rectangle
// found by the face detector, Stasm searches for the eyes in the
// appropriate subregions within the rectangle.  If both eyes are found the
// face is rotated so the eyes are horizontal.  The start shape is then
// formed by aligning the mean training shape to the eyes.  If either eye
// isn't found, the start shape is aligned to the face detector rectangle.
//
// Note however that if the eye angle is less than +-5 degrees, we treat it
// as 0 degrees (and don't rotate the face as described above).  This
// is to minimize preprocessing.
//
// 3. With the model estart=ESTART_EYE_AND_MOUTH (currently used for the
// three-quarter models), the start shape is generated as above, but we
// search for the mouth too and use it if is detected.
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "startshape.h"
#include "print.h"
#include "landmarks.h"
#include "stasm_landmarks.h"
#include "err.h"
#include "faceroi.h"
#include "eyedet.h"
#include "shape17.h"

namespace stasm
{

// Following used if we did not detect eyes.  We empirically get slighter better
// Stasm results if we slightly reduce the size of the detected face rectangle.

static const double FACERECT_SCALE_WHEN_NO_EYES = .95;

//-----------------------------------------------------------------------------

// Align meanshape to the face detector rectangle and return it as startshape
// This ignores the eye and mouth, if any.

static Shape AlignMeanShapeToFaceDet(
    const DetPar& detpar,                 // in
    const Shape&  meanshape,              // in
    double        scale,                  // in: scale the face rectangle
    const Image&  img)                    // io: the image (grayscale)
{
    if (trace_g)
        lprintf("AlignToFaceDet           ");

    DetPar detpar1(detpar);

    if (IsLeftFacing(detpar.eyaw))
        detpar1 = FlipDetPar(detpar, img.cols);

    CV_Assert(meanshape.rows > 0 && meanshape.cols == 2);

    const double xscale = detpar1.width  * scale / DET_FACE_WIDTH;
    const double yscale = detpar1.height * scale / DET_FACE_WIDTH;

    Shape startshape = TransformShape(meanshape,
                                     xscale,      0, detpar1.x,
                                          0, yscale, detpar1.y);

    return startshape;
}

// Return the model meanshape aligned to both eyes and the mouth.
//
// The central idea is to form a triangular shape of the eyes and
// bottom-of-mouth from the face detector params, and align the same
// triangle in the meanshape to this triangle.

static Shape AlignMeanShapeToBothEyesMouth(
    const DetPar& detpar,                   // in
    const Shape&  meanshape)                // in
{
    if (trace_g)
        lprintf("AlignToBothEyesMouth     ");

    CV_Assert(NSIZE(meanshape) > 0 && PointUsed(meanshape, 0));
    CV_Assert(Valid(detpar.mouthx));
    CV_Assert(Valid(detpar.lex));
    CV_Assert(Valid(detpar.rex));

    Shape mean_tri(3, 2), det_tri(3, 2);       // triangle of eyes and mouth

    const double x_meanmouth =
       (meanshape(L_CTopOfTopLip, IX) + meanshape(L_CBotOfBotLip, IX)) / 2.;

    const double y_meanmouth =
       (meanshape(L_CTopOfTopLip, IY) + meanshape(L_CBotOfBotLip, IY)) / 2.;

    const Shape shape17(Shape17(meanshape));

    mean_tri(0, IX) = shape17(L17_LPupil, IX); // left eye
    mean_tri(0, IY) = shape17(L17_LPupil, IY);
    mean_tri(1, IX) = shape17(L17_RPupil, IX); // right eye
    mean_tri(1, IY) = shape17(L17_RPupil, IY);
    mean_tri(2, IX) = x_meanmouth;             // mouth
    mean_tri(2, IY) = y_meanmouth;

    det_tri(0, IX) = detpar.lex;               // left eye
    det_tri(0, IY) = detpar.ley;
    det_tri(1, IX) = detpar.rex;               // right eye
    det_tri(1, IY) = detpar.rey;
    det_tri(2, IX) = detpar.mouthx;            // mouth
    det_tri(2, IY) = detpar.mouthy;

    return TransformShape(meanshape, AlignmentMat(mean_tri, det_tri));
}

// return the model meanshape aligned to both eyes (mouth is not avail)

static Shape AlignMeanShapeToBothEyesNoMouth(
    const DetPar& detpar,                      // in
    const Shape&  meanshape)                   // in
{
    if (trace_g)
        lprintf("AlignToBothEyesNoMouth   ");

    CV_Assert(NSIZE(meanshape) > 0 && PointUsed(meanshape, 0));
    CV_Assert(Valid(detpar.lex));
    CV_Assert(Valid(detpar.rex));

    Shape meanline(2, 2), detline(2, 2);       // line from eye to eye

    const Shape shape17(Shape17(meanshape));

    meanline(0, IX) = shape17(L17_LPupil, IX); // left eye
    meanline(0, IY) = shape17(L17_LPupil, IY);
    meanline(1, IX) = shape17(L17_RPupil, IX); // right eye
    meanline(1, IY) = shape17(L17_RPupil, IY);

    detline(0, IX) = detpar.lex;               // left eye
    detline(0, IY) = detpar.ley;
    detline(1, IX) = detpar.rex;               // right eye
    detline(1, IY) = detpar.rey;

    return TransformShape(meanshape, AlignmentMat(meanline, detline));
}

// return the model meanshape aligned to both eyes (mouth is not avail)

static Shape AlignMeanShapeToBothEyesEstMouth(
    const DetPar& detpar,                      // in
    const Shape&  meanshape)                   // in
{
    // .48 was tested to give slightly better worse case results than .50
    static const double EYEMOUTH_TO_FACERECT_RATIO = .48;

    if (trace_g)
        lprintf("AlignToBothEyesEstMouth  ");

    CV_Assert(NSIZE(meanshape) > 0 && PointUsed(meanshape, 0));
    CV_Assert(Valid(detpar.lex));
    CV_Assert(Valid(detpar.rex));

    // estimate the mouth's position

    double x_eyemid = 0;
    switch (detpar.eyaw)
    {
    case EYAW00:                                 //  mid point
        x_eyemid = .50 * detpar.lex + .50 * detpar.rex;
        break;
    // TODO The constants below have not been empirically optimized.
    case EYAW_45:                                // closer to left eye
        x_eyemid = .30 * detpar.lex + .70 * detpar.rex;
        break;
    case EYAW_22:                                // closer to left eye
        x_eyemid = .30 * detpar.lex + .70 * detpar.rex;
        break;
    case EYAW22:                                 // closer to right eye
        x_eyemid = .30 * detpar.lex + .70 * detpar.rex;
        break;
    case EYAW45:                                 // closer to right eye
        x_eyemid = .30 * detpar.lex + .70 * detpar.rex;
        break;
    default:
        Err("AlignMeanShapeToBothEyesEstMouth: Invalid eyaw %d", detpar.eyaw);
        break;
    }
    const double y_eyemid = (detpar.ley + detpar.rey) / 2;

    Shape mean_tri(3, 2), det_tri(3, 2);             // triangle of eyes and mouth

    const Shape shape17(Shape17(meanshape));

    mean_tri(0, IX) = shape17(L17_LPupil, IX);       // left eye
    mean_tri(0, IY) = shape17(L17_LPupil, IY);
    mean_tri(1, IX) = shape17(L17_RPupil, IX);       // right eye
    mean_tri(1, IY) = shape17(L17_RPupil, IY);
    mean_tri(2, IX) = shape17(L17_CBotOfBotLip, IX); // mouth
    mean_tri(2, IY) = shape17(L17_CBotOfBotLip, IY);

    det_tri(0, IX) = detpar.lex;                     // left eye
    det_tri(0, IY) = detpar.ley;
    det_tri(1, IX) = detpar.rex;                     // right eye
    det_tri(1, IY) = detpar.rey;
    det_tri(2, IX) = x_eyemid;                       // mouth
    det_tri(2, IY) = y_eyemid + EYEMOUTH_TO_FACERECT_RATIO * detpar.width;

    return TransformShape(meanshape, AlignmentMat(mean_tri, det_tri));
}

static Shape AlignMeanShapeToLeftEyeMouth(
    const DetPar& detpar,                             // in
    const Shape&  meanshape)                          // in
{
    if (trace_g)
        lprintf("AlignToLeftEyeMouth      ");

    CV_Assert(NSIZE(meanshape) > 0 && PointUsed(meanshape, 0));
    CV_Assert(Valid(detpar.lex));    // left eye valid?
    CV_Assert(!Valid(detpar.rex));   // right eye invalid? (else why are we here?)
    CV_Assert(Valid(detpar.mouthx)); // mouth valid?

    Shape meanline(2, 2), detline(2, 2);              // line from eye to mouth

    const Shape shape17(Shape17(meanshape));

    const double x_meanmouth =
       (shape17(L17_CTopOfTopLip, IX) + shape17(L17_CBotOfBotLip, IX)) / 2;

    const double y_meanmouth =
       (shape17(L17_CTopOfTopLip, IY) + shape17(L17_CBotOfBotLip, IY)) / 2;

    meanline(0, IX) = shape17(L17_LPupil, IX);        // left eye
    meanline(0, IY) = shape17(L17_LPupil, IY);
    meanline(1, IX) = x_meanmouth;                    // mouth
    meanline(1, IY) = y_meanmouth;

    detline(0, IX) = detpar.lex;                      // left eye
    detline(0, IY) = detpar.ley;
    detline(1, IX) = detpar.mouthx;                   // mouth
    detline(1, IY) = detpar.mouthy;

    return TransformShape(meanshape, AlignmentMat(meanline, detline));
}

static Shape AlignMeanShapeToRightEyeMouth(
    const DetPar& detpar,                             // in
    const Shape&  meanshape)                          // in
{
    if (trace_g)
        lprintf("AlignToRightEyeMouth     ");

    CV_Assert(NSIZE(meanshape) > 0 && PointUsed(meanshape, 0));
    CV_Assert(!Valid(detpar.lex));   // left eye invalid? (else why are we here?)
    CV_Assert(Valid(detpar.rex));    // right eye valid?
    CV_Assert(Valid(detpar.mouthx)); // mouth valid?

    const Shape shape17(Shape17(meanshape));

    const double x_meanmouth =
       (shape17(L17_CTopOfTopLip, IX) + shape17(L17_CBotOfBotLip, IX)) / 2;

    const double y_meanmouth =
       (shape17(L17_CTopOfTopLip, IY) + shape17(L17_CBotOfBotLip, IY)) / 2;

    Shape meanline(2, 2), detline(2, 2);              // line from eye to mouth

    meanline(0, IX) = shape17(L17_RPupil, IX);        // right eye
    meanline(0, IY) = shape17(L17_RPupil, IY);
    meanline(1, IX) = x_meanmouth;                    // mouth
    meanline(1, IY) = y_meanmouth;

    detline(0, IX) = detpar.rex;                      // right eye
    detline(0, IY) = detpar.rey;
    detline(1, IX) = detpar.mouthx;                   // mouth
    detline(1, IY) = detpar.mouthy;

    return TransformShape(meanshape, AlignmentMat(meanline, detline));
}

static void FlipIfLeftFacing(
    Shape& shape,             // io
    EYAW   eyaw,              // in
    int    ncols)             // in
{
    if (IsLeftFacing(eyaw))
        shape = FlipShape(shape, ncols);
}

static Shape EstartRectOnly(
    const DetPar& detpar_roi,      // in: detpar wrt the ROI
    const Image&  face_roi,        // in
    const Shape&  meanshape)       // in
{
    return AlignMeanShapeToFaceDet(detpar_roi, meanshape,
                                       FACERECT_SCALE_WHEN_NO_EYES, face_roi);
}

static Shape EstartEyes(
    const DetPar& detpar_roi,      // in: detpar wrt the ROI
    const Image&  face_roi,        // in
    const Shape&  meanshape)       // in
{
    Shape startshape;
    Shape meanshape1(meanshape);
    if (Valid(detpar_roi.lex) && Valid(detpar_roi.rex)) // both eyes available?
    {
        FlipIfLeftFacing(meanshape1, detpar_roi.eyaw, face_roi.cols);
        // TODO Tune the following code, what approach is best?
        if (detpar_roi.eyaw == EYAW00)
            startshape = AlignMeanShapeToBothEyesEstMouth(detpar_roi, meanshape1);
        else
            startshape = AlignMeanShapeToBothEyesNoMouth(detpar_roi, meanshape1);
        FlipIfLeftFacing(startshape, detpar_roi.eyaw, face_roi.cols);
    }
    else // at least one eye is missing, use the face det rectangle
    {
        startshape =
            AlignMeanShapeToFaceDet(detpar_roi, meanshape1,
                                        FACERECT_SCALE_WHEN_NO_EYES, face_roi);
    }
    return startshape;
}

static Shape EstartEyeMouth(
    const DetPar& detpar_roi,      // in: detpar wrt the ROI
    const Image&  face_roi,        // in
    const Shape&  meanshape)       // in
{
    Shape startshape;
    Shape meanshape1(meanshape);
    if (Valid(detpar_roi.mouthx) &&              // both eyes and mouth?
        Valid(detpar_roi.lex) &&
        Valid(detpar_roi.rex))
    {
        FlipIfLeftFacing(meanshape1, detpar_roi.eyaw, face_roi.cols);
        startshape = AlignMeanShapeToBothEyesMouth(detpar_roi, meanshape1);
        FlipIfLeftFacing(startshape, detpar_roi.eyaw, face_roi.cols);
    }
    else if (Valid(detpar_roi.lex) &&                 // both eyes?
             Valid(detpar_roi.rex))
    {
        FlipIfLeftFacing(meanshape1, detpar_roi.eyaw, face_roi.cols);
        // TODO Tune the following code, what approach is best?
        if (detpar_roi.eyaw == EYAW00)
            startshape = AlignMeanShapeToBothEyesEstMouth(detpar_roi, meanshape1);
        else
            startshape = AlignMeanShapeToBothEyesNoMouth(detpar_roi, meanshape1);
        FlipIfLeftFacing(startshape, detpar_roi.eyaw, face_roi.cols);
    }
    else if (Valid(detpar_roi.mouthx) &&         // left eye and mouth?
             Valid(detpar_roi.lex))
    {
        FlipIfLeftFacing(meanshape1, detpar_roi.eyaw, face_roi.cols);
        startshape = AlignMeanShapeToLeftEyeMouth(detpar_roi, meanshape1);
        FlipIfLeftFacing(startshape, detpar_roi.eyaw, face_roi.cols);
    }
    else if (Valid(detpar_roi.mouthx) &&         // right eye and mouth?
             Valid(detpar_roi.rex))
    {
        FlipIfLeftFacing(meanshape1, detpar_roi.eyaw, face_roi.cols);
        startshape = AlignMeanShapeToRightEyeMouth(detpar_roi, meanshape1);
        FlipIfLeftFacing(startshape, detpar_roi.eyaw, face_roi.cols);
    }
    else // last resort: use the face det rectangle (can't use facial features)
    {
        startshape =
            AlignMeanShapeToFaceDet(detpar_roi, meanshape1,
                                        FACERECT_SCALE_WHEN_NO_EYES, face_roi);
    }
    return startshape;
}

// Align the model meanshape to the detpar from the face and feature dets.
// Complexity enters in because the detected eyes and mouth may be useful
// if available.  The "left facing" code is needed because our three
// quarter models are for right facing faces (wrt the viewer).

static Shape StartShapeFromDetPar(
    const DetPar& detpar_roi,      // in: detpar wrt the ROI
    const Image&  face_roi,        // in
    const Shape&  meanshape,       // in
    ESTART        estart)          // in: use mouth etc. to posn start shape?
{
    Shape startshape;
    switch (estart)
    {
    case ESTART_RECT_ONLY:
        startshape = EstartRectOnly(detpar_roi, face_roi, meanshape);
        break;
    case ESTART_EYES:
        startshape = EstartEyes(detpar_roi, face_roi, meanshape);
        break;
    case ESTART_EYE_AND_MOUTH:
        startshape = EstartEyeMouth(detpar_roi, face_roi, meanshape);
        break;
    default:
        Err("Invalid estart %d", estart);
    }
    return JitterPointsAt00(startshape);
}

static void TraceEyesMouth(
    Image&  face_roi,           // out: ROI around face, possibly rotated upright
    DetPar& detpar_roi)         // out: detpar wrt to face_roi
{
    (void) face_roi;
    (void) detpar_roi;
#if TRACE_IMAGES // will be 0 unless debugging (defined in stasm.h)

    CImage cimg; cvtColor(face_roi, cimg, CV_GRAY2BGR); // color image
    rectangle(cimg,
              cv::Point(cvRound(detpar_roi.x - .5 * detpar_roi.width),
                        cvRound(detpar_roi.y - .5 * detpar_roi.height)),
              cv::Point(cvRound(detpar_roi.x + .5 * detpar_roi.width),
                        cvRound(detpar_roi.y + .5 * detpar_roi.height)),
              ToCvColor(C_BLUE), 3);
    if (Valid(detpar_roi.lex))
        cv::circle(cimg,
                   cv::Point(cvRound(detpar_roi.lex), cvRound(detpar_roi.ley)),
                   MAX(2, face_roi.cols / 40),
                   cv::Scalar(0, 0, 255), 2);
    if (Valid(detpar_roi.rex))
        cv::circle(cimg,
                   cv::Point(cvRound(detpar_roi.rex), cvRound(detpar_roi.rey)),
                   MAX(2, face_roi.cols / 40),
                   cv::Scalar(0, 0, 255), 2);
    if (Valid(detpar_roi.mouthx))
        cv::circle(cimg,
                   cv::Point(cvRound(detpar_roi.mouthx), cvRound(detpar_roi.mouthy)),
                   MAX(2, face_roi.cols / 40),
                   cv::Scalar(0, 0, 255), 2);
    char s[SLEN]; sprintf(s, "%s_25_eyemouth.bmp", Base(imgpath_g));
    lprintf("%s\n", s);
    if (!cv::imwrite(s, cimg))
        Err("Cannot write %s", s);

#endif
}

double EyeAngle(          // eye angle in degrees, INVALID if eye angle not available
    const DetPar& detpar) // in: detpar wrt the ROI
{
    double angle = 0;
    if (Valid(detpar.lex) && Valid(detpar.rey)) // both eyes detected?
    {
        angle = RadsToDegrees(
                    -atan2(detpar.rey - detpar.ley,
                           detpar.rex - detpar.lex));
    }
    return angle;
}

double EyeAngle(        // eye angle in degrees, INVALID if eye angle not available
    const Shape& shape) // in
{
    double angle = INVALID;

    const Shape shape17(Shape17OrEmpty(shape));

    if (shape17.rows && // converted shape to a Shape17 successfully?
        Valid(shape17(L17_LPupil, IX)) && Valid(shape17(L17_RPupil, IX)) &&
        PointUsed(shape17, L17_LPupil) && PointUsed(shape17, L17_RPupil))
    {
        angle = RadsToDegrees(
                    -atan2(shape17(L17_RPupil, IY) - shape17(L17_LPupil, IY),
                           shape17(L17_RPupil, IX) - shape17(L17_LPupil, IX)));
    }
    return angle;
}

// Get the start shape and the ROI around it, given the face rectangle.
// Depending on the estart field in the model, we detect the eyes
// and mouth and use those to help fit the start shape.
// (Note also that the ROI is flipped if necessary because our three-quarter
// models are right facing and the face may be left facing.)

static void StartShapeAndRoi(  // we have the facerect, now get the rest
    Shape&         startshape, // out: the start shape we are looking for
    Image&         face_roi,   // out: ROI around face, possibly rotated upright
    DetPar&        detpar_roi, // out: detpar wrt to face_roi
    DetPar&        detpar,     // io:  detpar wrt to img (has face rect on entry)
    const Image&   img,        // in:  the image (grayscale)
    const vec_Mod& mods,      // in:  a vector of models, one for each yaw range
                               //       (use only estart, and meanshape)
    StasmCascadeClassifier cascade)
{
    PossiblySetRotToZero(detpar.rot);          // treat small rots as zero rots

    FaceRoiAndDetPar(face_roi, detpar_roi,     // get ROI around face
                     img, detpar, false);

    DetectEyesAndMouth(detpar_roi,             // use OpenCV eye and mouth detectors
                       face_roi, cascade);

    // Some face detectors return the face rotation, some don't (in
    // the call to NextFace_ just made via NextStartShapeAndRoi).
    // If we don't have the rotation, then estimate it from the eye
    // angle, if the eyes are available.

    if (!Valid(detpar.rot)) // don't have the face rotation?
    {
        detpar_roi.rot = EyeAngle(detpar_roi);
        if (!Valid(detpar_roi.rot)) // eye angle not available?
            detpar_roi.rot = 0;
        PossiblySetRotToZero(detpar_roi.rot);
        detpar.rot = detpar_roi.rot;
        if (detpar.rot != 0)
        {
            // face is rotated: rotate ROI and re-get the eyes and mouth

            // TODO: Prevent bogus OpenCV assert fail face_roi.data == img.data.
            face_roi = Image(0,0);

            FaceRoiAndDetPar(face_roi, detpar_roi,
                             img, detpar, false);

            DetectEyesAndMouth(detpar_roi,    // use OpenCV eye and mouth detectors
                               face_roi, cascade);
        }
    }
    TraceEyesMouth(face_roi, detpar_roi);
    if (trace_g)
        lprintf("%-6.6s yaw %3.0f rot %3.0f ",
            EyawAsString(detpar_roi.eyaw), detpar_roi.yaw, detpar_roi.rot);
    else
        logprintf("%-6.6s yaw %3.0f rot %3.0f ",
            EyawAsString(detpar_roi.eyaw), detpar_roi.yaw, detpar_roi.rot);

    // select an ASM model based on the face's yaw
    const Mod* mod = mods[ABS(EyawAsModIndex(detpar_roi.eyaw, mods))];

    const ESTART estart = mod->Estart_();

    startshape = StartShapeFromDetPar(detpar_roi,
                                      face_roi, mod->MeanShape_(), estart);

    detpar.lex = detpar_roi.lex;

    if (IsLeftFacing(detpar_roi.eyaw))
        FlipImgInPlace(face_roi);

    JitterPointsAt00InPlace(startshape);
}

// Get the start shape for the next face in the image, and the ROI around it.
// The returned shape is wrt the ROI frame.
//
// Note that we we previously called the face detector, and the face
// rectangle(s) were saved privately in facedet, and are now ready for
// immediate retrieval by NextFace_.
//
// The following comment applies for three-quarter models (not for frontal
// models): If the three-quarter face is left-facing, we flip the ROI so
// the returned face is right-facing.  This is because our three-quarter
// ASM models are for right-facing faces.  For frontal faces (the yaw00
// model), faces are not flipped.

bool NextStartShapeAndRoi(     // use face detector results to estimate start shape
    Shape&         startshape, // out: the start shape
    Image&         face_roi,   // out: ROI around face, possibly rotated upright
    DetPar&        detpar_roi, // out: detpar wrt to face_roi
    DetPar&        detpar,     // out: detpar wrt to img
    const Image&   img,        // in:  the image (grayscale)
    const vec_Mod& mods,       // in:  a vector of models, one for each yaw range
                               //       (use only estart, and meanshape)
    FaceDet&       facedet,    // io:  the face detector (internal face index bumped)
    StasmCascadeClassifier cascade)
{
    detpar = facedet.NextFace_();  // get next face's detpar from the face det

    if (Valid(detpar.x))           // NextFace_ returned a face?
        StartShapeAndRoi(startshape, face_roi, detpar_roi, detpar, img, mods, cascade);

    return Valid(detpar.x);
}

} // namespace stasm
