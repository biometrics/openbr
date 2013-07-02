// faceroi.cpp: face ROI, and translation from image frame to the ROI
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"

namespace stasm
{
// Rotations less than 5 are treated as zero to minimize image preprocessing.

static const double ROT_TREAT_AS_ZERO = 5;

//-----------------------------------------------------------------------------

// Return a rect which covers the face with enough space around it for an
// ASM search, but also ensuring that the rect is in the image boundaries.

static Rect RoiRect(
    const DetPar& detpar,    // in
    int           nimgcols,  // in
    int           nimgrows,  // in
    bool          flip,      // in: mirror the ROI
    double        botfrac,   // in: distance from center to bottom marg
    double        leftfrac,  // in: dist from center to left marg
    double        topfrac,   // in
    double        rightfrac) // in
{
    int ixmin, ixmax;
    if (flip)
    {
        ixmin = MAX(0,        cvRound(detpar.x - rightfrac * detpar.width));
        ixmax = MIN(nimgcols, cvRound(detpar.x + leftfrac  * detpar.width));
    }
    else
    {
        ixmin = MAX(0,        cvRound(detpar.x - leftfrac  * detpar.width));
        ixmax = MIN(nimgcols, cvRound(detpar.x + rightfrac * detpar.width));
    }
    const int iymin = MAX(0,        cvRound(detpar.y - botfrac   * detpar.height));
    const int iymax = MIN(nimgrows, cvRound(detpar.y + topfrac   * detpar.height));

    Rect roi;

    roi.x = ixmin;
    roi.y = iymin;
    roi.width  = ixmax - ixmin;
    roi.height = iymax - iymin;

    CV_Assert(roi.width > 0);
    CV_Assert(roi.height > 0);

    return roi;
}

static bool IsRoiEntireImg(
    const Rect& roi,        // in
    int         imgcols,    // in
    int         imgrows)    // in
{
    return roi.x == 0 &&
           roi.y == 0 &&
           roi.width == imgcols &&
           roi.height == imgrows;
}

static DetPar ImgDetParToRoiFrame(
    const DetPar& detpar,          // in
    const Rect&   rect_roi)        // in
{
    DetPar detpar_roi(detpar);
    detpar_roi.x -= rect_roi.x;
    detpar_roi.y -= rect_roi.y;
    Shape eyemouth_shape(5, 2, 0.);
    if (Valid(detpar_roi.lex))
    {
        eyemouth_shape(0, IX) -= rect_roi.x;
        eyemouth_shape(0, IY) -= rect_roi.y;
    }
    if (Valid(detpar_roi.rex))
    {
        eyemouth_shape(1, IX) -= rect_roi.x;
        eyemouth_shape(1, IY) -= rect_roi.y;
    }
    if (Valid(detpar_roi.mouthx))
    {
        eyemouth_shape(2, IX) -= rect_roi.x;
        eyemouth_shape(2, IY) -= rect_roi.y;
    }
    if (Valid(detpar.rot) && detpar.rot)
    {
        // rotate eyes and mouth
        const MAT rotmat = getRotationMatrix2D(cv::Point2f(float(detpar_roi.x),
                                               float(detpar_roi.y)),
                                               -detpar.rot, 1.);
        AlignShapeInPlace(eyemouth_shape, rotmat);
    }
    if (Valid(detpar.lex))
    {
        detpar_roi.lex    = eyemouth_shape(0, IX);
        detpar_roi.ley    = eyemouth_shape(0, IY);
    }
    if (Valid(detpar.rex))
    {
        detpar_roi.rex    = eyemouth_shape(1, IX);
        detpar_roi.rey    = eyemouth_shape(1, IY);
    }
    if (Valid(detpar.mouthx))
    {
        detpar_roi.mouthx = eyemouth_shape(2, IX);
        detpar_roi.mouthy = eyemouth_shape(2, IY);
    }
    return detpar_roi;
}

Shape ImgShapeToRoiFrame(     // return shape in ROI frame
    const Shape&  shape,      // in: shape in image frame
    const DetPar& detpar_roi, // in: detpar wrt the ROI
    const DetPar& detpar)     // in
{
    Shape outshape(shape.clone());
    for (int i = 0; i < outshape.rows; i++)
        if (PointUsed(outshape, i))
        {
            outshape(i, IX) -= detpar.x - detpar_roi.x;
            outshape(i, IY) -= detpar.y - detpar_roi.y;
        }

    if (Valid(detpar.rot) && detpar.rot)
    {
        const MAT rotmat = getRotationMatrix2D(cv::Point2f(float(detpar_roi.x),
                                               float(detpar_roi.y)),
                                               -detpar.rot,
                                               1.);
        outshape = AlignShape(outshape, rotmat);
    }
    return outshape;
}

// In StartShapeAndRoi we selected a ROI and possibly rotated that ROI.
// The search was done on that ROI.  Now de-adjust the search results
// to undo the effects of searching on the ROI, not on the actual image.

Shape RoiShapeToImgFrame(     // return shape in image frame
    const Shape&  shape,      // in: shape in roi frame
    const Image&  face_roi,   // in
    const DetPar& detpar_roi, // in: detpar wrt the ROI
    const DetPar& detpar)     // in: detpar wrt the image
{
    Shape outshape(shape.clone());
    if (IsLeftFacing(detpar.eyaw))
        outshape = FlipShape(outshape, face_roi.cols);
    if (Valid(detpar.rot) && detpar.rot)
    {
        const MAT rotmat = getRotationMatrix2D(cv::Point2f(float(detpar_roi.x),
                                               float(detpar_roi.y)),
                                               detpar.rot, 1.);
        outshape = AlignShape(outshape, rotmat);
    }
    for (int i = 0; i < outshape.rows; i++)
        if (PointUsed(outshape, i))
        {
            outshape(i, IX) += detpar.x - detpar_roi.x;
            outshape(i, IY) += detpar.y - detpar_roi.y;
        }
    return outshape;
}

void PossiblySetRotToZero( // this is to avoid rotating the image unnecessarily
    double& rot)           // io
{
    if (rot >= -ROT_TREAT_AS_ZERO && rot <= ROT_TREAT_AS_ZERO)
        rot = 0;
}

void FaceRoiAndDetPar(        // get ROI around the face, rotate if necessary
    Image&        face_roi,   // out
    DetPar&       detpar_roi, // out: detpar wrt the ROI
    const Image&  img,        // in: original image
    const DetPar& detpar,     // in: wrt img frame, only x,y,w,h,rot used
    bool          flip,       // in: mirror the ROI?
    double        botfrac,    // in: default ROI_FRAC
    double        leftfrac,   // in: dist from center to left margin
    double        topfrac,    // in
    double        rightfrac)  // in
{
    Rect rect_roi = RoiRect(detpar, img.cols, img.rows, flip,
                            botfrac, leftfrac, topfrac,  rightfrac);

    detpar_roi = ImgDetParToRoiFrame(detpar, rect_roi);

    // following "if"s are for efficiency (avoid rotation etc. when possible).

    if (detpar.rot == 0 && IsRoiEntireImg(rect_roi, img.cols, img.rows))
        face_roi = img;

    else if (!Valid(detpar.rot) || detpar.rot == 0)
        face_roi = Image(img, rect_roi);

    else // rotate image so face is upright, results go into face_roi
        warpAffine(Image(img, rect_roi), face_roi,
                   getRotationMatrix2D(cv::Point2f(float(detpar_roi.x),
                                                   float(detpar_roi.y)),
                                       -detpar.rot, 1.),
                   cv::Size(face_roi.cols, face_roi.rows),
                   cv::INTER_AREA, cv::BORDER_REPLICATE);

    // TODO For efficiency could combine this flip with above rot img when possible?
    if (flip)
        FlipImgInPlace(face_roi);
}

} // namespace stasm
