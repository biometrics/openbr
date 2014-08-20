// shape17.cp: convert a shape to a 17 point shape
//
// We can conveniently work with shapes with differing numbers of points
// (e.g. XM2VTS, BioID) by first converting them to a "Shape17" shape.
// That is, we use Shape17s as a lowest common denominator.  A Shape17
// consists of the 17 points defined by Cristinacce's "me17" measure.

#include "stasm.h"

namespace stasm
{
static const double REF17[] = // mean frontal shape (data from mean MUCT shape)
{
    -44.2,  43.4,      // 0  L17_LPupil
     39.0,  48.1,      // 1  L17_RPupil
    -32.7, -41.6,      // 2  L17_LMouthCorner
     37.2, -37.7,      // 3  L17_RMouthCorner
    -74.6,  57.6,      // 4  L17_LEyebrowOuter
    -22.0,  63.3,      // 5  L17_LEyebrowInner
     14.8,  65.3,      // 6  L17_REyebrowInner
     67.6,  65.6,      // 7  L17_REyebrowOuter
    -59.2,  40.4,      // 8  L17_LEyeOuter
    -28.4,  42.0,      // 9  L17_LEyeInner
     23.6,  44.9,      // 10 L17_REyeInner
     54.3,  46.8,      // 11 L17_REyeOuter
     -0.2,   5.2,      // 12 L17_CNoseTip
    -13.8,  -4.3,      // 13 L17_LNostril
     14.4,  -2.7,      // 14 L17_RNostril
      1.6, -27.7,      // 15 L17_CTopOTopLip
      3.3, -56.3       // 16 L17_CBotOfBotLip
};

const Shape MEANSHAPE17(17, 2, const_cast<double*>(REF17));

//-----------------------------------------------------------------------------

static const int* Shape17Tab( // get appropriate tab to convert to a 17 point shape
    const Shape& shape)       // in
{
    static const int shape17_tab[17] = // identity transform
    {
         0, // 0  L17_LPupil
         1, // 1  L17_RPupil
         2, // 2  L17_LMouthCorner
         3, // 3  L17_RMouthCorner
         4, // 4  L17_LEyebrowOuter
         5, // 5  L17_LEyebrowInner
         6, // 6  L17_REyebrowInner
         7, // 7  L17_REyebrowOuter
         8, // 8  L17_LEyeOuter
         9, // 9  L17_LEyeInner
        10, // 10 L17_REyeInner
        11, // 11 L17_REyeOuter
        12, // 12 L17_CNoseTip
        13, // 13 L17_LNostril
        14, // 14 L17_RNostril
        15, // 15 L17_CTopOfTopLip
        16  // 16 L17_CBotOfBotLip
    };
    static const int bioid_tab[17] = // 20 points
    {
         0, // 0  L17_LPupil
         1, // 1  L17_RPupil
         2, // 2  L17_LMouthCorner
         3, // 3  L17_RMouthCorner
         4, // 4  L17_LEyebrowOuter
         5, // 5  L17_LEyebrowInner
         6, // 6  L17_REyebrowInner
         7, // 7  L17_REyebrowOuter
         9, // 8  L17_LEyeOuter
        10, // 9  L17_LEyeInner
        11, // 10 L17_REyeInner
        12, // 11 L17_REyeOuter
        14, // 12 L17_CNoseTip
        15, // 13 L17_LNostril
        16, // 14 L17_RNostril
        17, // 15 L17_CTopOfTopLip
        18  // 16 L17_CBotOfBotLip
    };
    static const int muct_tab[17] = // 68 (XM2VTS) or 76 (MUCT) points
    {
        31, // 0  L17_LPupil
        36, // 1  L17_RPupil
        48, // 2  L17_LMouthCorner
        54, // 3  L17_RMouthCorner
        21, // 4  L17_LEyebrowOuter
        24, // 5  L17_LEyebrowInner
        18, // 6  L17_REyebrowInner
        15, // 7  L17_REyebrowOuter
        27, // 8  L17_LEyeOuter
        29, // 9  L17_LEyeInner
        34, // 10 L17_REyeInner
        32, // 11 L17_REyeOuter
        67, // 12 L17_CNoseTip
        46, // 13 L17_LNostril
        47, // 14 L17_RNostril
        51, // 15 L17_CTopOfTopLip
        57  // 16 L17_CBotOfBotLip
    };
    static const int stasm77_tab[17] = // Stasm 77 points
    {
        38, // 0  LPupil
        39, // 1  RPupil
        59, // 2  LMouthCorner
        65, // 3  RMouthCorner
        18, // 4  LEyebrowOuter
        21, // 5  LEyebrowInner
        22, // 6  REyebrowInner
        25, // 7  REyebrowOuter
        34, // 8  LEyeOuter
        30, // 9  LEyeInner
        40, // 10 REyeInner
        44, // 11 REyeOuter
        52, // 12 CNoseTip
        51, // 13 LNostril
        53, // 14 RNostril
        62, // 15 CTopOfTopLip
        74  // 16 CBotOfBotLip
    };
    static const int helen_tab[17] = // 194 points (Helen)
    {
        144, // 0  LPupil actually eye outer corner, will correct in TweakHelen
        124, // 1  RPupil actually eye outer corner, will correct in TweakHelen
         58, // 2  LMouthCorner
         71, // 3  RMouthCorner
        185, // 4  LEyebrowOuter
        174, // 5  LEyebrowInner
        154, // 6  REyebrowInner
        164, // 7  REyebrowOuter
        144, // 8  LEyeOuter
        134, // 9  LEyeInner
        114, // 10 REyeInner
        124, // 11 REyeOuter
         49, // 12 CNoseTip actually base of nose, will correct in TweakHelen
         47, // 13 LNostril actually base of nostril, will correct in TweakHelen
         51, // 14 RNostril actually base of nostril, will correct in TweakHelen
         64, // 15 CTopOfTopLip
         79  // 16 CBotOfBotLip
        };
    static const int put199_tab[17] = // 199 points (extended PUT)
    {
        195, // 0  LPupil
        194, // 1  RPupil
         58, // 2  LMouthCorner
         72, // 3  RMouthCorner
        184, // 4  LEyebrowOuter
        174, // 5  LEyebrowInner
        154, // 6  REyebrowInner
        164, // 7  REyebrowOuter
        144, // 8  LEyeOuter
        134, // 9  LEyeInner
        114, // 10 REyeInner
        124, // 11 REyeOuter
        196, // 12 CNoseTip
        197, // 13 LNostril
        198, // 14 RNostril
         65, // 15 CTopOfTopLip
         79  // 16 CBotOfBotLip
    };
    const int *tab = NULL;
    switch(shape.rows)
    {
        case  17: tab = shape17_tab; break; // identity transform
        case  20: tab = bioid_tab;   break; // BioID
        case  22: tab = bioid_tab;   break; // AR
        case  68: tab = muct_tab;    break; // XM2VTS, 68 point MUCT
        case  76: tab = muct_tab;    break; // Stasm version 3 (Stasm76 shape)
        case  77: tab = stasm77_tab; break; // Stasm version 4 (Stasm77 shape)
        case 194: tab = helen_tab;   break; // Helen
        case 199: tab = put199_tab;  break; // PUT with me17 points
        default: Err("Cannot convert %d point shape to 17 points", shape.rows); break;
    }
    return tab;
}

static void TweakHelen(
    Shape& shape17)     // io
{
    // pupils not available in the helen set so use mean of eye corners
    if (PointUsed(shape17, L17_LEyeOuter) && PointUsed(shape17, L17_LEyeInner))
    {
        shape17(L17_LPupil, IX) = (shape17(L17_LEyeOuter, IX) + shape17(L17_LEyeInner, IX)) / 2;
        shape17(L17_LPupil, IY) = (shape17(L17_LEyeOuter, IY) + shape17(L17_LEyeInner, IY)) / 2;
    }
    if (PointUsed(shape17, L17_REyeOuter) && PointUsed(shape17, L17_REyeInner))
    {
        shape17(L17_RPupil, IX) = (shape17(L17_REyeOuter, IX) + shape17(L17_REyeInner, IX)) / 2;
        shape17(L17_RPupil, IY) = (shape17(L17_REyeOuter, IY) + shape17(L17_REyeInner, IY)) / 2;
    }
    // nose tip and nostrils not available, fake it by shifting available points up
    if (PointUsed(shape17, L17_LPupil) && PointUsed(shape17,  L17_RPupil))
    {
        const double shift = .08 * PointDist(shape17, L17_LPupil, L17_RPupil);
        shape17(L17_CNoseTip, IY) -= 2 * shift;
        shape17(L17_LNostril, IY) -= shift;
        shape17(L17_RNostril, IY) -= shift;
    }
}

Shape Shape17(          // convert an arb face shape to a 17 point shape
    const Shape& shape) // in
{
    const int* const tab = Shape17Tab(shape);
    Shape shape17(17, 2);
    for (int i = 0; i < 17; i++)
    {
        int iold = tab[i];
        CV_Assert(iold >= 0 && iold < NSIZE(shape));
        shape17(i, IX) = shape(iold, IX);
        shape17(i, IY) = shape(iold, IY);
    }
    if (shape.rows == 194) // helen set? www.ifp.illinois.edu/~vuongle2/helen
        TweakHelen(shape17);
    return shape17;
}

} // namespace stasm
