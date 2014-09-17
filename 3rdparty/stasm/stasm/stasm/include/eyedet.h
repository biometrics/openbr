// eyedet.h: interface to OpenCV eye and mouth detectors
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_EYEDET_H
#define STASM_EYEDET_H

#include "misc.h"
#include "asm.h"
#include "stasmcascadeclassifier.h"

namespace stasm
{
bool NeedEyes(                 // true if we need the eye detectors for the given mods
    const vec_Mod& mods);      // in: the ASM model(s)

bool NeedMouth(                // true if we need the mouth detector for the given mods
    const vec_Mod& mods);      // in: the ASM model(s)

void OpenEyeMouthDetectors(    // open eye and mouth detectors, if necessary
    bool           need_eyes,  // in: true if we need the eye detectors
    bool           need_mouth, // in: true if we need the mouth detector
    const char*    datadir);   // in

void OpenEyeMouthDetectors(  // possibly open OpenCV eye detectors and mouth detector
    const vec_Mod& mods,     // in: the ASM models (used to see if we need eyes or mouth)
    const char*    datadir); // in

void DetectEyesAndMouth(     // use OpenCV detectors to find the eyes and mouth
    DetPar&      detpar,     // io: eye and mouth fields updated, other fields untouched
    const Image& img,       // in: ROI around face (already rotated if necessary)
    StasmCascadeClassifier cascade);

} // namespace stasm
#endif // STASM_EYEDET_H
