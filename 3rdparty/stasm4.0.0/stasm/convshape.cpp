// convshape.cpp: convert a stasm 77 point shape to other formats
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"

namespace stasm
{
static void Copy(          // copy a point from oldshape to shape
    Shape&       shape,    // io
    const Shape& oldshape, // in
    int          i,        // in
    int          iold)     // in
{
    shape(i, IX) = oldshape(iold, IX);
    shape(i, IY) = oldshape(iold, IY);
}

static void Inter(         // interpolate a point from two nearby oldshape points
    Shape&       shape,    // io
    const Shape& oldshape, // in
    int          i,        // in: shape point
    double       ratio,    // in: interpolation ratio, 0 to 1
    int          i1,       // in: oldshape point 1
    int          i2)       // in: oldshape point 2
{
    if (!PointUsed(oldshape, i1) && !PointUsed(oldshape, i2))
    {
        shape(i, IX) = 0;
        shape(i, IY) = 0;
    }
    else if (!PointUsed(oldshape, i1))
    {
        shape(i, IX) = oldshape(i2, IX) + 1; // +1 is arb, to disambiguate point
        shape(i, IY) = oldshape(i2, IY) + 1;
    }
    else if (!PointUsed(oldshape, i2))
    {
        shape(i, IX) = oldshape(i1, IX) + 1;
        shape(i, IY) = oldshape(i1, IY) + 1;
    }
    else
    {
        CV_Assert(ratio >= 0 && ratio <= 1);
        shape(i, IX) = ratio * oldshape(i1, IX) + (1-ratio) * oldshape(i2, IX);
        shape(i, IY) = ratio * oldshape(i1, IY) + (1-ratio) * oldshape(i2, IY);
    }
}

static Shape Shape77As20( // return an approximated BioID 20 point shape
    const Shape& shape)   // in: Stasm 77 point shape
{
    CV_Assert(shape.rows == 77);

    Shape newshape(20, 2);

    Copy(newshape, shape,  0, 38);
    Copy(newshape, shape,  1, 39);
    Copy(newshape, shape,  2, 59);
    Copy(newshape, shape,  3, 65);
    Copy(newshape, shape,  4, 18);
    Copy(newshape, shape,  5, 21);
    Copy(newshape, shape,  6, 22);
    Copy(newshape, shape,  7, 25);
    Copy(newshape, shape,  8, 0);
    Copy(newshape, shape,  9, 34);
    Copy(newshape, shape, 10, 30);
    Copy(newshape, shape, 11, 40);
    Copy(newshape, shape, 12, 44);
    Copy(newshape, shape, 13, 12);
    Copy(newshape, shape, 14, 52);
    Copy(newshape, shape, 15, 51);
    Copy(newshape, shape, 16, 53);
    Copy(newshape, shape, 17, 62);
    Copy(newshape, shape, 18, 74);
    Copy(newshape, shape, 19, 6);

#if MOD_A1 || MOD_A || MOD_A_EMU
    const double eyemouth = EyeMouthDist(shape);
    newshape(15, IY) += MAX(1, .02 * eyemouth); // move down, into nostril
    newshape(16, IY) += MAX(1, .02 * eyemouth); // move down, into nostril
#endif

    return newshape;
}

static Shape Shape77As22( // return an approximated AR 22 point shape
    const Shape& shape)   // in: Stasm 77 point shape
{
    CV_Assert(shape.rows == 77);

    // first 20 points same as BioId
    Shape newshape = DimKeep(Shape77As20(shape), 77, 2);

    Copy(newshape, shape, 20, 3);
    Copy(newshape, shape, 21, 9);

    return newshape;
}

static Shape Shape77As68( // return an approximated XM2VTS 68 point shape
    const Shape& shape)   // in: Stasm 77 point shape
{
    CV_Assert(shape.rows == 77);

    Shape newshape(68, 2);

    Copy(newshape,  shape,  0,  0);
    Inter(newshape, shape,  1, .6667, 1, 2);
    Inter(newshape, shape,  2, .5,    2, 3);
    Copy(newshape,  shape,  3,  3);
    Inter(newshape, shape,  4, .3333, 3, 4);
    Inter(newshape, shape,  5, .6667, 4, 5);
    Copy(newshape,  shape,  6,  5);
    Copy(newshape,  shape,  7,  6);
    Copy(newshape,  shape,  8,  7);
    Inter(newshape, shape,  9, .3333, 7, 8);
    Inter(newshape, shape, 10, .6667, 8, 9);
    Copy(newshape,  shape, 11,  9);
    Inter(newshape, shape, 12, .5,    9, 10);
    Inter(newshape, shape, 13, .3333, 10, 11);
    Copy(newshape,  shape, 14, 12);
    Copy(newshape,  shape, 15, 25);
    Copy(newshape,  shape, 16, 24);
    Copy(newshape,  shape, 17, 23);
    Copy(newshape,  shape, 18, 22);
    Copy(newshape,  shape, 19, 27);
    Copy(newshape,  shape, 20, 26);
    Copy(newshape,  shape, 21, 18);
    Copy(newshape,  shape, 22, 17);
    Copy(newshape,  shape, 23, 16);
    Copy(newshape,  shape, 24, 21);
    Copy(newshape,  shape, 25, 20);
    Copy(newshape,  shape, 26, 19);
    Copy(newshape,  shape, 27, 34);
    Copy(newshape,  shape, 28, 32);
    Copy(newshape,  shape, 29, 30);
    Copy(newshape,  shape, 30, 36);
    Copy(newshape,  shape, 31, 38);
    Copy(newshape,  shape, 32, 44);
    Copy(newshape,  shape, 33, 42);
    Copy(newshape,  shape, 34, 40);
    Copy(newshape,  shape, 35, 46);
    Copy(newshape,  shape, 36, 39);
    Inter(newshape, shape, 37, .6667, 30, 40);
    newshape(37, IX) = shape(50, IX);
    Copy(newshape,  shape, 38, 50);
    Copy(newshape,  shape, 39, 58);
    Copy(newshape,  shape, 40, 57);
    Copy(newshape,  shape, 41, 56);
    Copy(newshape,  shape, 42, 55);
    Copy(newshape,  shape, 43, 54);
    Copy(newshape,  shape, 44, 48);
    Inter(newshape, shape, 45, .3333, 30, 40);
    newshape(45, IX) = shape(48, IX);
    Copy(newshape,  shape, 46, 51);
    Copy(newshape,  shape, 47, 53);
    Copy(newshape,  shape, 48, 59);
    Copy(newshape,  shape, 49, 60);
    Copy(newshape,  shape, 50, 61);
    Copy(newshape,  shape, 51, 62);
    Copy(newshape,  shape, 52, 63);
    Copy(newshape,  shape, 53, 64);
    Copy(newshape,  shape, 54, 65);
    Copy(newshape,  shape, 55, 72);
    Copy(newshape,  shape, 56, 73);
    Copy(newshape,  shape, 57, 74);
    Copy(newshape,  shape, 58, 75);
    Copy(newshape,  shape, 59, 76);
    Copy(newshape,  shape, 60, 69);
    Copy(newshape,  shape, 61, 70);
    Copy(newshape,  shape, 62, 71);
    Copy(newshape,  shape, 63, 66);
    Copy(newshape,  shape, 64, 67);
    Copy(newshape,  shape, 65, 68);
    Inter(newshape, shape, 66, .5, 67, 70);
    Copy(newshape,  shape, 67, 52);

#if MOD_A1 || MOD_A || MOD_A_EMU
    const double eyemouth = EyeMouthDist(shape);
    newshape(38, IY) += MAX(1, .05 * eyemouth); // move side of nose down
    newshape(44, IY) += MAX(1, .05 * eyemouth); // move side of nose down
    newshape(46, IY) += MAX(1, .02 * eyemouth); // move down, into nostril
    newshape(47, IY) += MAX(1, .02 * eyemouth); // move down, into nostril
#endif

    return newshape;
}

static Shape Shape77As76( // return an approximated MUCT 76 point shape
    const Shape& shape)   // in: Stasm 77 point shape
{
    // first 68 points same as XM2VTS
    Shape newshape = DimKeep(Shape77As68(shape), 77, 2);

    Copy(newshape,  shape, 68, 33); // extra eyelid points
    Copy(newshape,  shape, 69, 31);
    Copy(newshape,  shape, 70, 37);
    Copy(newshape,  shape, 71, 35);
    Copy(newshape,  shape, 72, 43);
    Copy(newshape,  shape, 73, 41);
    Copy(newshape,  shape, 74, 47);
    Copy(newshape,  shape, 75, 45);

    return newshape;
}

Shape ConvertShape(          // return shape with nlandmarks, return no rows if can't
    const Shape& shape,      // in
    int          nlandmarks) // in: 77=nochange, 76=stasm3, 68=xm2vts, 22=ar, 20=bioid, 17=me17
{
    Shape newshape;
    if (nlandmarks == shape.rows)
        newshape = shape.clone();
    else if (nlandmarks == 17) // me17
        newshape = Shape17(shape);
    else if (shape.rows == stasm_NLANDMARKS)
    {
        switch(nlandmarks)
        {
            case 20: newshape = Shape77As20(shape); break; // BioID
            case 22: newshape = Shape77As22(shape); break; // AR
            case 68: newshape = Shape77As68(shape); break; // XM2VTS
            case 76: newshape = Shape77As76(shape); break; // Stasm 3
        }
    }
    return newshape;
}

} // namespace stasm
