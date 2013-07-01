// eyedist.cpp: calculate eye-mouth and inter-eye dist
//
// The functions in this file know how to deal with missing points.  This
// matters during testing when we are comparing results to manually
// landmarked reference shapes.  For example, a reference eye pupil may be
// concealed by the side of the nose.  When calculating the inter-eye
// distance, if the pupil is missing we can instead use a point near the
// pupil.  We must then adjust the point-to-point distance calculated using
// this surrogate point.  We use the mean face shape to figure out the
// adjustment.  The accuracy of the resulting estimated inter-eye distance will
// depend upon how similar the proportions of the face are to the mean face.
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"

namespace stasm
{
static int TabPoint(    // return first used point in tab, -1 if none
    const int*   tab,   // in
    int          ntab,  // in
    const Shape& shape) // in
{
    for (int i = 0; i < ntab; i++)
        if (PointUsed(shape, tab[i]))
            return tab[i]; // note return

    return -1;
}

// TODO Use center of mouth rather than bottom of bottom lip
//      --- but then would have to to retrain the ASM models.

static double CanonicalEyeMouthDist( // return 0 if pupils and mouth not avail
    const Shape& shape17) // in
{
    if (!PointUsed(shape17, L17_LPupil) ||
        !PointUsed(shape17, L17_RPupil) ||
        !PointUsed(shape17, L17_CBotOfBotLip))
    {
        return 0; // note return
    }
    return PointDist(
             MeanPoint(shape17, L17_LPupil, L17_RPupil, IX), // eye mid point
             MeanPoint(shape17, L17_LPupil, L17_RPupil, IY),
             shape17(L17_CBotOfBotLip, IX),                  // bot of bot lip
             shape17(L17_CBotOfBotLip, IY));
}

double EyeMouthDist(    // eye-mouth distance of a face shape
    const Shape& shape) // in
{
    static const int eyes[] = // surrogates for pupil midpoint
    {
        L17_LPupil,
        L17_RPupil,
        L17_LEyeOuter,
        L17_REyeOuter,
        L17_LEyeInner,
        L17_REyeInner,
        L17_LEyebrowInner,
        L17_REyebrowInner,
        L17_LEyebrowOuter,
        L17_REyebrowOuter
    };
    static const int mouths[] = // surrogates for bot of bot lip
    {
        L17_CBotOfBotLip,
        L17_CTopOfTopLip,
        L17_LMouthCorner,
        L17_RMouthCorner
    };
    const Shape shape17(shape.rows == 17? shape: Shape17(shape));
    double eyemouth = CanonicalEyeMouthDist(shape17);
    if (eyemouth == 0) // pupils and mouth not available?
    {
        const int eye   = TabPoint(eyes,   NELEMS(eyes),   shape17);
        const int mouth = TabPoint(mouths, NELEMS(mouths), shape17);
        if (eye >= 0 && mouth >= 0)  // actual or surrogate points available?
        {
            eyemouth = PointDist(shape17, eye, mouth) *
                       CanonicalEyeMouthDist(MEANSHAPE17) /
                       PointDist(MEANSHAPE17, eye, mouth);
        }
    }
    if (eyemouth == 0)
    {
        // last resort, estimate eyemouth dist from shape extent
        eyemouth = MAX(ShapeWidth(shape17), ShapeHeight(shape17)) *
                   PointDist(MEANSHAPE17, L17_LPupil, L17_CBotOfBotLip) /
                   MAX(ShapeWidth(MEANSHAPE17), ShapeHeight(MEANSHAPE17));
    }
    CV_Assert(eyemouth > 1 && eyemouth < 1e5); // sanity check
    return eyemouth;
}

double InterEyeDist(    // inter-pupil distance of a face shape
    const Shape& shape) // in
{
    static const int leyes[] = // surrogates for left pupil
    {
        L17_LPupil,
        L17_LEyeOuter,
        L17_LEyeInner,
        L17_LEyebrowInner,
        L17_LEyebrowOuter
    };
    static const int reyes[] = // surrogates for right pupil
    {
        L17_RPupil,
        L17_REyeOuter,
        L17_REyeInner,
        L17_REyebrowInner,
        L17_REyebrowOuter
    };
    double eyedist = 0;
    const Shape shape17(Shape17(shape));
    const int leye = TabPoint(leyes, NELEMS(leyes), shape17);
    const int reye = TabPoint(reyes, NELEMS(reyes), shape17);
    if (leye >= 0 && reye >= 0) // actual or surrogate points available?
    {
        eyedist = PointDist(shape17, leye, reye) *
                  PointDist(MEANSHAPE17, L17_LPupil, L17_RPupil) /
                  PointDist(MEANSHAPE17, leye, reye);
    }
    else // last resort, estimate inter-pupil distance from shape extent
    {
        eyedist = MAX(ShapeWidth(shape17), ShapeHeight(shape17)) *
                  PointDist(MEANSHAPE17, L17_LPupil, L17_RPupil) /
                  MAX(ShapeWidth(MEANSHAPE17), ShapeHeight(MEANSHAPE17));
    }
    CV_Assert(eyedist > 1 && eyedist < 1e5); // sanity check
    return eyedist;
}

} // namespace stasm
