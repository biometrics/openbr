// pinstart.h: utilities for creating a start shape from manually pinned points
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_PINSTART_H
#define STASM_PINSTART_H

namespace stasm
{

void PinnedStartShapeAndRoi(   // use the pinned landmarks to init the start shape
    Shape&         startshape, // out: the start shape (in ROI frame)
    Image&         face_roi,   // out: ROI around face, possibly rotated upright
    DetPar&        detpar_roi, // out: detpar wrt to face_roi
    DetPar&        detpar,     // out: detpar wrt to img
    Shape&         pinned_roi, // out: pinned arg translated to ROI frame
    const Image&   img,        // in: the image (grayscale)
    const vec_Mod& mods,       // in: a vector of models, one for each yaw range
    const Shape&   pinned);    // in: manually pinned landmarks

} // namespace stasm
#endif // STASM_PINSTART_H
