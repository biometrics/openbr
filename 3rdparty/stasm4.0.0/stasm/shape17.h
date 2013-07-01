// shape17.h: convert a shape to a 17 point shape
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_SHAPE17_H
#define STASM_SHAPE17_H

namespace stasm
{
enum LANDMARKS_17       // the 17 points that make up a Shape17 shape
{
    L17_LPupil,        //  0
    L17_RPupil,        //  1
    L17_LMouthCorner,  //  2
    L17_RMouthCorner,  //  3
    L17_LEyebrowOuter, //  4
    L17_LEyebrowInner, //  5
    L17_REyebrowInner, //  6
    L17_REyebrowOuter, //  7
    L17_LEyeOuter,     //  8
    L17_LEyeInner,     //  9
    L17_REyeInner,     // 10
    L17_REyeOuter,     // 11
    L17_CNoseTip,      // 12
    L17_LNostril,      // 13
    L17_RNostril,      // 14
    L17_CTopOfTopLip,  // 15
    L17_CBotOfBotLip   // 16
};

#if 0
static const char *LANDMARKS_17_NAMES[] =
{
    "LPupil",
    "RPupil",
    "LMouthCorner",
    "RMouthCorner",
    "LEyebrowOuter",
    "LEyebrowInner",
    "REyebrowInner",
    "REyebrowOuter",
    "LEyeOuter",
    "LEyeInner",
    "REyeInner",
    "REyeOuter",
    "CNoseTip",
    "LNostril",
    "RNostril",
    "CTopOTopLip",
    "CBotOfBotLip"
};
#endif

extern const Shape MEANSHAPE17; // mean 17 point shape

Shape Shape17(           // convert an arb face shape to a 17 point shape
    const Shape& shape); // in

} // namespace stasm
#endif // STASM_SHAPE17_H
