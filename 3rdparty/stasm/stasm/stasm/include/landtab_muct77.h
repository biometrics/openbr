// landtab_muct77.h: definitions for MUCT 77 point shapes
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_LANDTAB_MUCT77_H
#define STASM_LANDTAB_MUCT77_H

#include "atface.h"

// Note that the AT_Hat bit is ignored if pyr lev > HAT_START_LEV

static const LANDMARK_INFO LANDMARK_INFO_TAB[77] = // stasm77 points
{
//   par pre next weight bits
    { 12,  1, 30,  1., AT_Beard|AT_Glasses      }, // 00 L_LTemple
    { 11, -1, -1,  1., AT_Beard|AT_Glasses      }, // 01 L_LJaw01
    { 10, -1, -1,  1., AT_Beard                 }, // 02 L_LJawNoseline
    {  9, -1, -1,  1., AT_Beard                 }, // 03 L_LJawMouthline
    {  8, -1, -1,  1., AT_Beard                 }, // 04 L_LJaw04
    {  7, -1, -1,  1., AT_Beard                 }, // 05 L_LJaw05
    { -1, -1, -1,  1., AT_Beard                 }, // 06 L_CTipOfChin
    {  5, -1, -1,  1., AT_Beard                 }, // 07 L_RJaw07
    {  4, -1, -1,  1., AT_Beard                 }, // 08 L_RJaw08
    {  3, -1, -1,  1., AT_Beard                 }, // 09 L_RJawMouthline
    {  2, -1, -1,  1., AT_Beard                 }, // 10 L_RJawNoseline
    {  1, -1, -1,  1., AT_Beard|AT_Glasses      }, // 11 L_RJaw11
    {  0, 11, 40,  1., AT_Beard|AT_Glasses      }, // 12 L_RTemple
    { 15, 12, 14,  0., 0                        }, // 13 L_RForehead point is virtually useless
    { -1, -1, -1,  0., 0                        }, // 14 L_CForehead point is virtually useless
    { 13,  0, 14,  0., 0                        }, // 15 L_LForehead point is virtually useless
    { 23, 17, 21,  0., AT_Glasses|AT_Hat        }, // 16 L_LEyebrowTopInner
    { 24, -1, -1,  0., AT_Glasses|AT_Hat        }, // 17 L_LEyebrowTopOuter
    { 25,  0, 17,  0., AT_Glasses|AT_Hat        }, // 18 L_LEyebrowOuter
    { 26, -1, -1,  0., AT_Glasses|AT_Hat        }, // 19 L_LEyebrowBotOuter
    { 27, 19, 21,  0., AT_Glasses|AT_Hat        }, // 20 L_LEyebrowBotInner
    { 22,  1, 11,  0., AT_Glasses|AT_Hat        }, // 21 L_LEyebrowInner
    { 21,  1, 11,  0., AT_Glasses|AT_Hat        }, // 22 L_REyebrowInner
    { 16, 22, 24,  0., AT_Glasses|AT_Hat        }, // 23 L_REyebrowTopInner
    { 17, -1, -1,  0., AT_Glasses|AT_Hat        }, // 24 L_REyebrowTopOuter
    { 18, 12, 24,  0., AT_Glasses|AT_Hat        }, // 25 L_REyebrowOuter
    { 19, -1, -1,  0., AT_Glasses|AT_Hat        }, // 26 L_REyebrowBotOuter
    { 20, 22, 26,  0., AT_Glasses|AT_Hat        }, // 27 L_REyebrowBotInner
    { 29, 26, 39,  1., AT_Glasses|AT_Eye|AT_Hat }, // 28 L_REyelid
    { 28, 19, 38,  1., AT_Glasses|AT_Eye|AT_Hat }, // 29 L_LEyelid
    { 40, 32, 36,  1., AT_Glasses|AT_Eye|AT_Hat }, // 30 L_LEyeInner
    { 41, -1, -1,  1., AT_Glasses|AT_Eye|AT_Hat }, // 31 L_LEye31
    { 42, -1, -1,  1., AT_Glasses|AT_Eye|AT_Hat }, // 32 L_LEyeTop
    { 43, -1, -1,  1., AT_Glasses|AT_Eye|AT_Hat }, // 33 L_LEye33
    { 44, 32, 36,  1., AT_Glasses|AT_Eye|AT_Hat }, // 34 L_LEyeOuter
    { 45, -1, -1,  1., AT_Glasses|AT_Eye|AT_Hat }, // 35 L_LEye35
    { 46, -1, -1,  1., AT_Glasses|AT_Eye|AT_Hat }, // 36 L_LEyeBot
    { 47, 30, 36,  1., AT_Glasses|AT_Eye|AT_Hat }, // 37 L_LEye37
    { 39, 34, 30,  1., AT_Glasses|AT_Eye|AT_Hat }, // 38 L_LPupil
    { 38, 44, 40,  1., AT_Glasses|AT_Eye|AT_Hat }, // 39 L_RPupil
    { 30, 42, 46,  1., AT_Glasses|AT_Eye|AT_Hat }, // 40 L_REyeInner
    { 31, -1, -1,  1., AT_Glasses|AT_Eye|AT_Hat }, // 41 L_REye41
    { 32, -1, -1,  1., AT_Glasses|AT_Eye|AT_Hat }, // 42 L_REyeTop
    { 33, -1, -1,  1., AT_Glasses|AT_Eye|AT_Hat }, // 43 L_REye43
    { 34, 42, 46,  1., AT_Glasses|AT_Eye|AT_Hat }, // 44 L_REyeOuter
    { 35, -1, -1,  1., AT_Glasses|AT_Eye|AT_Hat }, // 45 L_REye45
    { 36, -1, -1,  1., AT_Glasses|AT_Eye|AT_Hat }, // 46 L_REyeBot
    { 37, 40, 46,  1., AT_Glasses|AT_Eye|AT_Hat }, // 47 L_REye47
    { 50, 40, 62,  1., AT_Glasses|AT_Hat        }, // 48 L_RNoseMid
    { -1,  1, 11,  1., AT_Glasses|AT_Hat        }, // 49 L_CNoseMid
    { 48, 30, 62,  1., AT_Glasses|AT_Hat        }, // 50 L_LNoseMid
    { 53,  1, 11,  1., AT_Hat                   }, // 51 L_LNostrilTop
    { -1,  1, 11,  1., AT_Hat                   }, // 52 L_CNoseTip
    { 51,  1, 11,  1., AT_Hat                   }, // 53 L_RNostrilTop
    { 58, 40, 62,  1., AT_Hat                   }, // 54 L_RNoseSide
    { 57,  1, 11,  1., AT_Hat                   }, // 55 L_RNostrilBot
    { -1, -1, -1,  1., AT_Hat                   }, // 56 L_CNoseBase
    { 55,  1, 11,  1., AT_Mustache|AT_Hat       }, // 57 L_LNostrilBot
    { 54, 30, 62,  1., AT_Mustache|AT_Hat       }, // 58 L_LNoseSide
    { 65, 61, 74,  1., AT_Mustache|AT_Hat       }, // 59 L_LMouthCorner
    { 64, 59, 61,  1., AT_Mustache|AT_Hat       }, // 60 L_LMouth60
    { 63, -1, -1,  1., AT_Mustache|AT_Hat       }, // 61 L_LMouthCupid
    { -1, -1, -1,  1., AT_Mustache|AT_Hat       }, // 62 L_CTopOfTopLip
    { 61, -1, -1,  1., AT_Mustache|AT_Hat       }, // 63 L_RMouthCupid
    { 60, 63, 65,  1., AT_Mustache|AT_Hat       }, // 64 L_RMouth64
    { 59, 63, 74,  1., AT_Mustache|AT_Hat       }, // 65 L_RMouthCorner
    { 68, 65, 67,  1., AT_Hat                   }, // 66 L_RMouth66
    { -1, -1, -1,  1., AT_Hat                   }, // 67 L_CBotOfTopLip
    { 66, 59, 67,  1., AT_Hat                   }, // 68 L_LMouth68
    { 71, 59, 70,  1., AT_Hat                   }, // 69 L_LMouth69
    { -1, -1, -1,  1., AT_Hat                   }, // 70 L_CTopOfBotLip
    { 69, 65, 70,  1., AT_Hat                   }, // 71 L_RMouth71
    { 76, 65, 73,  1., AT_Hat                   }, // 72 L_RMouth72
    { 75, 72, 74,  1., AT_Hat                   }, // 73 L_RMouth73
    { -1, -1, -1,  1., AT_Hat                   }, // 74 L_CBotOfBotLip
    { 73, 74, 76,  1., AT_Hat                   }, // 75 L_LMouth75
    { 72, 59, 75,  1., AT_Hat                   }, // 76 L_LMouth76
};

#endif // STASM_LANDTAB_MUCT77_H
