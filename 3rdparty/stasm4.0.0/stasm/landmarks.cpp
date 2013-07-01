// landmarks.cpp: code for manipulating landmarks
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"

namespace stasm
{
double MeanPoint(
    const Shape& shape,   // in
    int          ipoint1, // in
    int          ipoint2, // in
    int          ix)      // in: IX or IY
{
    return (shape(ipoint1, ix) + shape(ipoint2, ix)) / 2;
}

void PrevAndNextLandmarks(
    int&         prev,        // out
    int&         next,        // out
    int          ipoint,      // in
    const Shape& shape)       // in
{
    const int npoints = shape.rows;

    const LANDMARK_INFO* const info = LANDMARK_INFO_TAB;

    prev = info[ipoint].prev;
    if (prev < 0) // not specified in table?
        prev = (ipoint + npoints - 1) % npoints;

    next = info[ipoint].next;
    if (next < 0)
        next = (ipoint + 1) % npoints;

    CV_Assert(prev >= 0);
    CV_Assert(next >= 0);
    CV_Assert(prev != next);
    CV_Assert(PointUsed(shape, prev));
    CV_Assert(PointUsed(shape, next));
}

static void FlipPoint(
    Shape&       shape,    // io
    const Shape& oldshape, // in
    int          inew,     // in
    int          iold,     // in
    int          imgwidth) // in
{
    if (!PointUsed(oldshape, iold))
        shape(inew, IX) = shape(inew, IY) = 0;
    else
    {
        shape(inew, IX) = imgwidth - oldshape(iold, IX) - 1;
        shape(inew, IY) = oldshape(iold, IY);
        if (!PointUsed(shape, inew))   // falsely marked unused after conversion?
            shape(inew, IX) = XJITTER; // adjust so not marked as unused
    }
}

// Flip shape horizontally.
// Needed so we can use right facing  models for left facing faces.

Shape FlipShape(
    const Shape& shape,    // in
    int          imgwidth) // in
{
    const LANDMARK_INFO* info = LANDMARK_INFO_TAB;
    Shape outshape(shape.rows, 2);
    for (int i = 0; i < shape.rows; i++)
    {
        int partner = info[i].partner;

        if (partner == -1) // e.g. tip of nose
            partner = i;

        FlipPoint(outshape, shape, partner, i, imgwidth);
    }
    return outshape;
}

} // namespace stasm
