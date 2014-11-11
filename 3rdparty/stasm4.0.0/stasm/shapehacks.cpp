// shapehacks.cpp:
//
// The shape model sometimes allows implausible point layouts.  For
// example, the mouth on the nose, or the chin inside the mouth.  The
// functions in this module fix the most egregious cases.  These hacks
// don't necessarily make the overall fitness measure (FM29) better,
// but minimize the occurrence of ridiculous shapes, although also
// occasionally worsen a good shape.
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"

namespace stasm
{
static double SHIFT_MOUTH_FROM_NOSE_FRAC = 0.06; // .06 from tuning on D1 set

static double CHIN_DOWN_RATIO = 0.5; // chin must be this far from mouth
static double CHIN_DOWN_SHIFT = 0.2;

static double CHIN_UP_RATIO   = 2.4; // chin cannot be further than this from mouth
static double CHIN_UP_SHIFT   = 0.1;

static double TEMPLE_RATIO = .1; // temple must be this far from eye, 0 disables
static double TEMPLE_SHIFT = 3;

//-----------------------------------------------------------------------------

static void PossiblyPrint(const char* s) // debugging print
{
    if (trace_g)
        lprintf("%s ", s);
}

void ApplyShapeModelHacks( // adjust shape by applying various hacks
    Shape&   shape,        // io: position of features possibly adjusted
    unsigned hackbits)     // in: which hacks to apply, see SHAPEHACKS defs
{
    CV_Assert(shape.rows == stasm_NLANDMARKS); // the hacks assume stasm77 points
    CV_Assert(shape.rows == 77);

    const double eyemouth = EyeMouthDist(shape);

    if (hackbits & SHAPEHACKS_DEFAULT)
    {
        // Possibly shift the entire mouth down, if it is too close to the nose.
        // Useful when the descriptor matchers think the nostrils are the mouth.

        const double nosemouth_gap =
            shape(L_CTopOfTopLip, IY) - shape(L_CNoseBase, IY);
        if (nosemouth_gap < .1 * eyemouth)
        {
            PossiblyPrint("ShiftMouthDown");
            for (int i = L_LMouthCorner; i <= L_LMouth76; i++)
                shape(i, IY) += SHIFT_MOUTH_FROM_NOSE_FRAC * eyemouth;
        }
        // Shift the bottom of mouth down if it is above the top of mouth.

        const double gap = shape(L_CTopOfBotLip, IY) - shape(L_CTopOfTopLip, IY);
        if (gap < 0)
        {
            PossiblyPrint("ShiftBottomOfMouthDown");
            for (int i = L_RMouthCorner; i <= L_LMouth76; i++)
                shape(i, IY) -= gap;
        }
        // Possibly shift the chin down or up, if it too close to the mouth.
        // Useful when the chin is on the mouth.

        const double y_mouth_center =
           (shape(L_CTopOfTopLip, IY) + shape(L_CBotOfBotLip, IY)) / 2;
        const double nosemouth_gap1 =
            MAX(0, y_mouth_center - shape(L_CNoseBase, IY));
        const double mouthchin_gap =
            shape(L_CTipOfChin, IY) - y_mouth_center;
        if (mouthchin_gap < CHIN_DOWN_RATIO * nosemouth_gap1)
        {
            PossiblyPrint("ShiftChinDown");
            double yadjust = CHIN_DOWN_SHIFT * eyemouth;
            shape(L_LJaw04,     IY) += yadjust;
            shape(L_LJaw05,     IY) += yadjust;
            shape(L_CTipOfChin, IY) += yadjust;
            shape(L_RJaw07,     IY) += yadjust;
            shape(L_RJaw08,     IY) += yadjust;
        }
        if (mouthchin_gap > CHIN_UP_RATIO * nosemouth_gap1)
        {
            PossiblyPrint("ShiftChinUp");
            double yadjust = CHIN_UP_SHIFT * eyemouth;
            shape(L_LJaw04,     IY) -= yadjust;
            shape(L_LJaw05,     IY) -= yadjust;
            shape(L_CTipOfChin, IY) -= yadjust;
            shape(L_RJaw07,     IY) -= yadjust;
            shape(L_RJaw08,     IY) -= yadjust;
        }
    }
    // Possibly shift the side of face away from eye.
    // Useful when the side of face is on the eye.

    if (hackbits & SHAPEHACKS_SHIFT_TEMPLE_OUT)
    {
        if (shape(L_LTemple, IX) >
            shape(L_LEyeOuter, IX) - TEMPLE_RATIO * eyemouth)
        {
            PossiblyPrint("LTempleOut");
            double xadjust =
                TEMPLE_SHIFT * ABS(shape(L_LEyeOuter, IX) - shape(L_LTemple, IX));
            shape(L_LTemple,       IX) -= xadjust;
            shape(L_LJaw01,        IX) -= xadjust;
            shape(L_LJawNoseline,  IX) -= xadjust;
            shape(L_LJawMouthline, IX) -= .5 * xadjust;
        }
        if (shape(L_RTemple, IX) <
            shape(L_REyeOuter, IX) + TEMPLE_RATIO * eyemouth)
        {
            PossiblyPrint("RTempleOut");
            double xadjust =
                TEMPLE_SHIFT * ABS(shape(L_REyeOuter, IX) - shape(L_RTemple, IX));
            shape(L_RTemple,       IX) += xadjust;
            shape(L_RJaw11,        IX) += xadjust;
            shape(L_RJawNoseline,  IX) += xadjust;
            shape(L_RJawMouthline, IX) += .5 * xadjust;
        }
    }
}

} // namespace stasm
