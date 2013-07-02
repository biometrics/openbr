// startshape.h: routines for finding the start shape for an ASM search
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_startshape_H
#define STASM_startshape_H

#include "stasmcascadeclassifier.h"

namespace stasm
{
// get the start shape for the next face in the image, and the ROI around it

bool NextStartShapeAndRoi(     // use face detector results to estimate start shape
    Shape&         startshape, // out: the start shape we are looking for
    Image&         face_roi,   // out: ROI around face, possibly rotated upright
    DetPar&        detpar_roi, // out: detpar wrt to face_roi
    DetPar&        detpar,     // out: detpar wrt to img
    const Image&   img,        // in: the image (grayscale)
    const vec_Mod& mods,       // in: a vector of models, one for each yaw range
    FaceDet&       facedet,   // io:  the face detector (internal face index bumped)
    StasmCascadeClassifier cascade);

void PinnedStartShapeAndRoi(   // use the pinned landmarks to init the start shape
    Shape&         startshape, // out: the start shape (in ROI frame)
    Image&         face_roi,   // out: ROI around face, possibly rotated upright
    DetPar&        detpar_roi, // out: detpar wrt to face_roi
    DetPar&        detpar,     // out: detpar wrt to img
    Shape&         pinned_roi, // out: pinned arg translated to ROI frame
    const Image&   img,        // in: the image (grayscale)
    const vec_Mod& mods,       // in: a vector of models, one for each yaw range
    const Shape&   pinned);    // in: pinned landmarks

} // namespace stasm
#endif // STASM_startshape_H
