// faceroi.h: face ROI, and translation from image frame to the ROI
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_FACEROI_H
#define STASM_FACEROI_H

namespace stasm
{
// RoiFrac controls the size of the ROI area around the face.  So for
// example leftfrac is the distance from the center to the left margin in
// units of the original detector rect width.  So ROI_FRAC=.5, say, would
// return the original face detector rectangle.

static const double ROI_FRAC = 1.0; // ROI is double the face detector width

Shape ImgShapeToRoiFrame(     // return shape in ROI frame
    const Shape&  shape,      // in: shape in image frame
    const DetPar& detpar_roi, // in: detpar wrt the ROI
    const DetPar& detpar);    // in

Shape RoiShapeToImgFrame(     // return shape in image frame
    const Shape&  shape,      // in: shape in roi frame
    const Image&  face_roi,   // in
    const DetPar& detpar_roi, // in: detpar wrt the ROI
    const DetPar& detpar);    // in: detpar wrt the image

void PossiblySetRotToZero(    // avoid rotating the image unnecessarily
    double& rot);             // io

void FaceRoiAndDetPar(        // extract ROI around the face, rotate if necessary
    Image&        face_roi,   // out
    DetPar&       detpar_roi, // out: detpar wrt the ROI
    const Image&  img,        // in: original image
    const DetPar& detpar,     // in: wrt img frame, only x,y,w,h,rot used
    bool          flip,       // in: mirror the ROI
    double botfrac   = ROI_FRAC,  // in: distance from center to bottom marg
    double leftfrac  = ROI_FRAC,  // in: dist from center to left marg
    double topfrac   = ROI_FRAC,  // in
    double rightfrac = ROI_FRAC); // in

} // namespace stasm
#endif // STASM_FACEROI_H
