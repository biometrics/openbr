// landmarks.h: code for manipulating landmarks
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_LANDMARKS_H
#define STASM_LANDMARKS_H

namespace stasm
{
struct LANDMARK_INFO  // landmark information
{
    int   partner;    // symmetrical partner point, -1 means no partner

    int   prev, next; // previous and next point
                      // special val -1 means prev=current-1 and next=current+1

    double weight;    // weight of landmark relative to others (for shape mod)
};

static const LANDMARK_INFO LANDMARK_INFO_TAB[stasm_NLANDMARKS] = // stasm77 points
{
    //  par pre next weight
    {    12,  1, 15,  1. }, // 00 L_LTemple
    {    11, -1, -1,  1. }, // 01 L_LJaw01
    {    10, -1, -1,  1. }, // 02 L_LJawNoseline
    {     9, -1, -1,  1. }, // 03 L_LJawMouthline
    {     8, -1, -1,  1. }, // 04 L_LJaw04
    {     7, -1, -1,  1. }, // 05 L_LJaw05
    {    -1, -1, -1,  1. }, // 06 L_CTipOfChin
    {     5, -1, -1,  1. }, // 07 L_RJaw07
    {     4, -1, -1,  1. }, // 08 L_RJaw08
    {     3, -1, -1,  1. }, // 09 L_RJawMouthline
    {     2, -1, -1,  1. }, // 10 L_RJawNoseline
    {     1, -1, -1,  1. }, // 11 L_RJaw11
    {     0, 11, 13,  1. }, // 12 L_RTemple
    {    15, -1, -1,  0. }, // 13 L_RForehead point is virtually useless
    {    -1, -1, -1,  0. }, // 14 L_CForehead point is virtually useless
    {    13, -1, -1,  0. }, // 15 L_LForehead point is virtually useless
    {    23, -1, -1,  0. }, // 16 L_LEyebrowTopInner
    {    24, -1, -1,  0. }, // 17 L_LEyebrowTopOuter
    {    25,  0, 17,  0. }, // 18 L_LEyebrowOuter
    {    26, -1, -1,  0. }, // 19 L_LEyebrowBotOuter
    {    27, -1, -1,  0. }, // 20 L_LEyebrowBotInner
    {    22,  0, 12,  0. }, // 21 L_LEyebrowInner
    {    21,  0, 12,  0. }, // 22 L_REyebrowInner
    {    16, -1, -1,  0. }, // 23 L_REyebrowTopInner
    {    17, -1, -1,  0. }, // 24 L_REyebrowTopOuter
    {    18, 12, 24,  0. }, // 25 L_REyebrowOuter
    {    19, -1, -1,  0. }, // 26 L_REyebrowBotOuter
    {    20, -1, -1,  0. }, // 27 L_REyebrowBotInner
    {    29, 26, 39,  1. }, // 28 L_REyelid
    {    28, 20, 38,  1. }, // 29 L_LEyelid
    {    40, 32, 36,  1. }, // 30 L_LEyeInner
    {    41, -1, -1,  1. }, // 31 L_LEye31
    {    42, -1, -1,  1. }, // 32 L_LEyeTop
    {    43, -1, -1,  1. }, // 33 L_LEye33
    {    44, 32, 36,  1. }, // 34 L_LEyeOuter
    {    45, -1, -1,  1. }, // 35 L_LEye35
    {    46, -1, -1,  1. }, // 36 L_LEyeBot
    {    47, 30, 36,  1. }, // 37 L_LEye37
    {    39, -1, -1,  1. }, // 38 L_LPupil
    {    38, -1, -1,  1. }, // 39 L_RPupil
    {    30, 42, 46,  1. }, // 40 L_REyeInner
    {    31, -1, -1,  1. }, // 41 L_REye41
    {    32, -1, -1,  1. }, // 42 L_REyeTop
    {    33, -1, -1,  1. }, // 43 L_REye43
    {    34, 42, 46,  1. }, // 44 L_REyeOuter
    {    35, -1, -1,  1. }, // 45 L_REye45
    {    36, -1, -1,  1. }, // 46 L_REyeBot
    {    37, 40, 46,  1. }, // 47 L_REye47
    {    50,  0, 12,  1. }, // 48 L_RNoseMid
    {    -1, -1, -1,  1. }, // 49 L_CNoseMid
    {    48,  0, 12,  1. }, // 50 L_LNoseMid
    {    53,  0, 12,  1. }, // 51 L_LNostrilTop
    {    -1,  0, 12,  1. }, // 52 L_CNoseTip
    {    51,  0, 12,  1. }, // 53 L_RNostrilTop
    {    58,  0, 12,  1. }, // 54 L_RNoseSide
    {    57, 60, 62,  1. }, // 55 L_RNostrilBot
    {    -1, -1, -1,  1. }, // 56 L_CNoseBase
    {    55, 30, 62,  1. }, // 57 L_LNostrilBot
    {    54,  0, 12,  1. }, // 58 L_LNoseSide
    {    65, 61, 74,  1. }, // 59 L_LMouthCorner
    {    64, 59, 61,  1. }, // 60 L_LMouth60
    {    63, -1, -1,  1. }, // 61 L_LMouthCupid
    {    -1, -1, -1,  1. }, // 62 L_CTopOfTopLip
    {    61, -1, -1,  1. }, // 63 L_RMouthCupid
    {    60, 63, 65,  1. }, // 64 L_RMouth64
    {    59, 61, 74,  1. }, // 65 L_RMouthCorner
    {    68, 65, 67,  1. }, // 66 L_RMouth66
    {    -1, -1, -1,  1. }, // 67 L_CBotOfTopLip
    {    66, 59, 67,  1. }, // 68 L_LMouth68
    {    71, 59, 70,  1. }, // 69 L_LMouth69
    {    -1, -1, -1,  1. }, // 70 L_CTopOfBotLip
    {    69, 65, 70,  1. }, // 71 L_RMouth71
    {    76, 65, 73,  1. }, // 72 L_RMouth72
    {    75, 72, 74,  1. }, // 73 L_RMouth73
    {    -1, -1, -1,  1. }, // 74 L_CBotOfBotLip
    {    73, 74, 76,  1. }, // 75 L_LMouth75
    {    72, 59, 75,  1. }, // 76 L_LMouth76
};

double MeanPoint(
    const Shape& shape,       // in
    int          ipoint1,     // in
    int          ipoint2,     // in
    int          ix);         // in: IX or IY

void PrevAndNextLandmarks(
    int&         prev,        // out
    int&         next,        // out
    int          ipoint,      // in
    const Shape& shape);      // in

Shape FlipShape(              // flip shape horizontally
    const Shape& shape,       // in
    int          imgwidth);   // in

} // namespace stasm
#endif // STASM_LANDMARKS_H
