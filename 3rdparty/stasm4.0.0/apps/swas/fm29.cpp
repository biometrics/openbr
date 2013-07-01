// fm29.cpp: calculate the FM29 measure of fitness
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"
#include "fm29.h"

namespace stasm
{
// xres and yres are x,y resolutions.  A small value means the fit must be
// more accurate at that point in that direction.  Points with zero xres
// entries are ignored.

typedef struct FitParams // how each landmark affects overall fitness
    {
    double xres;
    double yres;
    }
FitParams;

#define _ 0

static const FitParams FITPARAMS[stasm_NLANDMARKS] =
{
    { _, _,   }, // 00 L_LTemple
    { _, _,   }, // 01 L_LJaw01
    { 2, 5,   }, // 02 L_LJawNoseline
    { 5, 5,   }, // 03 L_LJawMouthline
    { _, _,   }, // 04 L_LJaw04
    { 5, 5,   }, // 05 L_LJaw05
    { 5, 2,   }, // 06 L_CTipOfChin
    { 5, 5,   }, // 07 L_RJaw07
    { _, _,   }, // 08 L_RJaw08
    { 5, 5,   }, // 05 L_RJawMouthline
    { 2, 5,   }, // 10 L_RJawNoseline
    { _, _,   }, // 11 L_RJaw11
    { _, _,   }, // 12 L_RTemple
    { _, _,   }, // 13 L_RForehead
    { _, _,   }, // 14 L_CForehead
    { _, _,   }, // 15 L_LForehead
    { _, _,   }, // 16 L_LEyebrowTopInner
    { _, _,   }, // 17 L_LEyebrowTopOuter
    { _, _,   }, // 18 L_LEyebrowOuter
    { _, _,   }, // 19 L_LEyebrowBotOuter
    { _, _,   }, // 20 L_LEyebrowBotInner
    { _, _,   }, // 21 L_LEyebrowInner
    { _, _,   }, // 22 L_REyebrowInner
    { _, _,   }, // 23 L_REyebrowTopInner
    { _, _,   }, // 24 L_REyebrowTopOuter
    { _, _,   }, // 25 L_REyebrowOuter
    { _, _,   }, // 26 L_REyebrowBotOuter
    { _, _,   }, // 27 L_REyebrowBotInner
    { _, _,   }, // 28 L_REyelid
    { _, _,   }, // 29 L_LEyelid
    { 1, 1,   }, // 30 L_LEyeInner
    { 3, 1,   }, // 31 L_LEye31
    { _, _,   }, // 32 L_LEyeTop
    { 3, 1,   }, // 33 L_LEye33
    { 1, 1,   }, // 34 L_LEyeOuter
    { 3, 1,   }, // 35 L_LEye35
    { _, _,   }, // 36 L_LEyeBot
    { 3, 1,   }, // 37 L_LEye37
    { _, _,   }, // 38 L_LPupil
    { _, _,   }, // 39 L_RPupil
    { 1, 1,   }, // 40 L_REyeInner
    { 3, 1,   }, // 41 L_REye41
    { _, _,   }, // 42 L_REyeTop
    { 3, 1,   }, // 43 L_REye43
    { 1, 1,   }, // 44 L_REyeOuter
    { 3, 1,   }, // 45 L_REye45
    { _, _,   }, // 46 L_REyeBot
    { 3, 1,   }, // 47 L_REye47
    { _, _,   }, // 48 L_RNoseMid
    { _, _,   }, // 49 L_CNoseMid
    { _, _,   }, // 50 L_LNoseMid
    { 3, 3,   }, // 51 L_LNostrilTop
    { 3, 3,   }, // 52 L_CNoseTip
    { 3, 3,   }, // 53 L_RNostrilTop
    { _, _,   }, // 54 L_RNoseSide
    { _, _,   }, // 55 L_RNostrilBot
    { 3, 3,   }, // 56 L_CNoseBase
    { _, _,   }, // 57 L_LNostrilBot
    { _, _,   }, // 58 L_LNoseSide
    { 3, 1,   }, // 59 L_LMouthCorner
    { _, _,   }, // 60 L_LMouth60
    { _, _,   }, // 61 L_LMouthCupid
    { 3, 1,   }, // 62 L_CTopOfTopLip
    { _, _,   }, // 63 L_RMouthCupid
    { _, _,   }, // 64 L_RMouth64
    { 3, 1,   }, // 65 L_RMouthCorner
    { _, _,   }, // 66 L_RMouth66
    { 3, 1,   }, // 67 L_CBotOfTopLip
    { _, _,   }, // 68 L_LMouth68
    { _, _,   }, // 69 L_LMouth69
    { 3, 1,   }, // 70 L_CTopOfBotLip
    { _, _,   }, // 71 L_RMouth71
    { _, _,   }, // 72 L_RMouth72
    { _, _,   }, // 73 L_RMouth73
    { 3, 1,   }, // 74 L_CBotOfBotLip
    { _, _,   }, // 75 L_LMouth75
    { _, _, }    // 76 L_LMouth76
};

#undef _

//-----------------------------------------------------------------------------

void Fm29(
    double&      fm29,     // out: FM29 measure of fitness
    int&         iworst,   // out: index of point with worse fit
    const Shape& shape,    // in
    const Shape& refshape) // in
{
    if (shape.rows != 77)
        Err("Fitness measure FM29 can be used only on shapes with 77 points "
            "(your shape has %d points)", shape.rows);
    if (refshape.rows != 77)
        Err("Fitness measure FM29 can be used only on shapes with 77 points "
            "(your reference shape has %d points)", refshape.rows);

    fm29 = 0;
    iworst = -1;
    double worst = -1;
    double weight = 0;
    for (int i = 0; i < shape.rows; i++)
        if (FITPARAMS[i].xres &&    // point is used for FM29?
            PointUsed(refshape, i)) // point present in ref shape?
        {
            CV_Assert(PointUsed(shape, i));
            CV_Assert(FITPARAMS[i].yres);

            const double pointfit =
                SQ((shape(i, IX) - refshape(i, IX)) / FITPARAMS[i].xres) +
                SQ((shape(i, IY) - refshape(i, IY)) / FITPARAMS[i].yres);

            fm29 += pointfit;

            const double pointweight =
                1 / SQ(FITPARAMS[i].xres) +
                1 / SQ(FITPARAMS[i].yres);

            weight += pointweight;

            if (pointfit > worst)
            {
                worst = pointfit;
                iworst = i;
            }
        }

    CV_Assert(weight > 0);
    // multiply by 2 so same as mean euclidean dist when all xres = yres = 1
    fm29 = sqrt(fm29 * 2 / weight) / EyeMouthDist(refshape);
}

} // namespace stasm
