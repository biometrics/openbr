// stasm_lib.h: interface to the Stasm library
//
// The Stasm library interface function and variable names are all prefixed
// by "stasm_".  They are not in the stasm namespace.
//
// The library routines return 1 on success, 0 on error (they do not throw
// exceptions).  Use stasm_lasterr to get the error string.  An example
// error is "Cannot open ../data/haarcascade_frontalface_alt2.xml".
// Errors in OpenCV routines called by Stasm and out-of-memory errors are
// handled in the same way.
//
// Typical usage in multiple face scenario with stasm_search_auto:
//
//      stasm_init()
//      load image from disk
//      optionally present the image to the user
//      stasm_open_image()
//      while stasm_find_face() finds another face:
//          save the face shape (and optionally present the image to the user)
//      optionally present the image with all face shapes to the user
//      optionally call stasm_search_pinned to correct any bad faces
//
// The interface is defined in vanilla C so can be used by code
// in "any" language.
//
//-----------------------------------------------------------------------------
//
//               Stasm License Agreement
//
// Copyright (C) 2005-2013, Stephen Milborrow
// All rights reserved.
//
// Redistribution of Stasm in source and binary forms, with or
// without modification, is permitted provided that the following
// conditions are met:
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimers.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimers in the documentation
//     and/or other materials provided with the distribution.
//
// A SIFT patent restriction may be in conflict with the copyright
// freedoms granted by this license.  This license does not give you
// permission to infringe patents.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holder be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// Note that Stasm versions prior to version 4.0.0 had a different license.
//
//-----------------------------------------------------------------------------

#ifndef STASM_LIB_H
#define STASM_LIB_H

#include "stasmcascadeclassifier.h"

static const int stasm_NLANDMARKS = 77; // number of landmarks

extern const char* const stasm_VERSION;

extern "C"
int stasm_init(              // call once, at bootup
    const char*  datadir,    // in: directory of face detector files
    int          trace);     // in: 0 normal use, 1 trace to stdout and stasm.log

extern "C"
int stasm_open_image(        // call once per image, detect faces
    const char*  img,        // in: gray image data, top left corner at 0,0
    int          width,      // in: image width
    int          height,     // in: image height
    const char*  imgpath,    // in: image path, used only for err msgs and debug
    int          multiface,  // in: 0=return only one face, 1=allow multiple faces
    int          minwidth);  // in: min face width as percentage of img width

extern "C"
int stasm_search_auto(       // call repeatedly to find all faces
    int*         foundface,  // out: 0=no more faces, 1=found face
    float*       landmarks,  // out: x0, y0, x1, y1, ..., caller must allocate
    const char*  data,
    const int    width,
    const int    height,
    StasmCascadeClassifier cascade);

extern "C"
int stasm_search_single(     // wrapper for stasm_search_auto and friends
    int*         foundface,  // out: 0=no face, 1=found face
    float*       landmarks,  // out: x0, y0, x1, y1, ..., caller must allocate
    const char*  img,        // in: gray image data, top left corner at 0,0
    int          width,      // in: image width
    int          height,     // in: image height
    StasmCascadeClassifier cascade,
    const char*  imgpath,    // in: image path, used only for err msgs and debug
    const char*  datadir);   // in: directory of face detector files

extern "C"                   // find landmarks, no OpenCV face detect
int stasm_search_pinned(     // call after the user has pinned some points
    float*       landmarks,  // out: x0, y0, x1, y1, ..., caller must allocate
    const float* pinned,     // in: pinned landmarks (0,0 points not pinned)
    const char*  img,        // in: gray image data, top left corner at 0,0
    int          width,      // in: image width
    int          height,     // in: image height
    const char*  imgpath);   // in: image path, used only for err msgs and debug

extern "C"
const char* stasm_lasterr(void); // return string describing last error

extern "C"
void stasm_force_points_into_image( // force landmarks into image boundary
    float*       landmarks,         // io
    int          ncols,             // in
    int          nrows);            // in

extern "C"
void stasm_convert_shape( // convert stasm 77 points to given number of points
    float* landmarks,     // io: return all points zero if can't do conversion
    int    nlandmarks);   // in: 77=nochange 76=stasm3 68=xm2vts 22=ar 20=bioid 17=me17

// stasm_printf is like printf but also prints to the file stasm.log if it
// is open.  The file stasm.log will be open if stasm_init was called with
// trace=1.  This function was added primarily for the programs that test
// the stasm library.

extern "C"
void stasm_printf(const char* format, ...); // print to stdout and stasm.log

#endif // STASM_LIB_H
