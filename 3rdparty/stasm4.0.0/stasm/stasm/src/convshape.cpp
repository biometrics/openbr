// convshape.cpp: convert a shape such as stasm 77 point shape to other formats
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm_lib.h"
#include "convshape.h"
#include "shape17.h"

namespace stasm
{
static void CopyPoint(     // copy a point from oldshape to shape
    Shape&       shape,    // io
    const Shape& oldshape, // in
    int          i,        // in
    int          iold)     // in
{
    shape(i, IX) = oldshape(iold, IX);
    shape(i, IY) = oldshape(iold, IY);
}

static void InterPoint(    // interpolate a point from two nearby oldshape points
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

    CopyPoint(newshape, shape,  0, 38);
    CopyPoint(newshape, shape,  1, 39);
    CopyPoint(newshape, shape,  2, 59);
    CopyPoint(newshape, shape,  3, 65);
    CopyPoint(newshape, shape,  4, 18);
    CopyPoint(newshape, shape,  5, 21);
    CopyPoint(newshape, shape,  6, 22);
    CopyPoint(newshape, shape,  7, 25);
    CopyPoint(newshape, shape,  8, 0);
    CopyPoint(newshape, shape,  9, 34);
    CopyPoint(newshape, shape, 10, 30);
    CopyPoint(newshape, shape, 11, 40);
    CopyPoint(newshape, shape, 12, 44);
    CopyPoint(newshape, shape, 13, 12);
    CopyPoint(newshape, shape, 14, 52);
    CopyPoint(newshape, shape, 15, 51);
    CopyPoint(newshape, shape, 16, 53);
    CopyPoint(newshape, shape, 17, 62);
    CopyPoint(newshape, shape, 18, 74);
    CopyPoint(newshape, shape, 19, 6);

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
    Shape newshape = DimKeep(Shape77As20(shape), 22, 2);

    CopyPoint(newshape, shape, 20, 3);
    CopyPoint(newshape, shape, 21, 9);

    return newshape;
}

static Shape Shape77AsXm2vts68( // return an approximated XM2VTS 68 point shape
    const Shape& shape)         // in: Stasm 77 point shape
{
    CV_Assert(shape.rows == 77);

    Shape newshape(68, 2);

    CopyPoint(newshape,  shape,  0,  0);
    InterPoint(newshape, shape,  1, .6667, 1, 2);
    InterPoint(newshape, shape,  2, .5,    2, 3);
    CopyPoint(newshape,  shape,  3,  3);
    InterPoint(newshape, shape,  4, .3333, 3, 4);
    InterPoint(newshape, shape,  5, .6667, 4, 5);
    CopyPoint(newshape,  shape,  6,  5);
    CopyPoint(newshape,  shape,  7,  6);
    CopyPoint(newshape,  shape,  8,  7);
    InterPoint(newshape, shape,  9, .3333, 7, 8);
    InterPoint(newshape, shape, 10, .6667, 8, 9);
    CopyPoint(newshape,  shape, 11,  9);
    InterPoint(newshape, shape, 12, .5,    9, 10);
    InterPoint(newshape, shape, 13, .3333, 10, 11);
    CopyPoint(newshape,  shape, 14, 12);
    CopyPoint(newshape,  shape, 15, 25);
    CopyPoint(newshape,  shape, 16, 24);
    CopyPoint(newshape,  shape, 17, 23);
    CopyPoint(newshape,  shape, 18, 22);
    CopyPoint(newshape,  shape, 19, 27);
    CopyPoint(newshape,  shape, 20, 26);
    CopyPoint(newshape,  shape, 21, 18);
    CopyPoint(newshape,  shape, 22, 17);
    CopyPoint(newshape,  shape, 23, 16);
    CopyPoint(newshape,  shape, 24, 21);
    CopyPoint(newshape,  shape, 25, 20);
    CopyPoint(newshape,  shape, 26, 19);
    CopyPoint(newshape,  shape, 27, 34);
    CopyPoint(newshape,  shape, 28, 32);
    CopyPoint(newshape,  shape, 29, 30);
    CopyPoint(newshape,  shape, 30, 36);
    CopyPoint(newshape,  shape, 31, 38);
    CopyPoint(newshape,  shape, 32, 44);
    CopyPoint(newshape,  shape, 33, 42);
    CopyPoint(newshape,  shape, 34, 40);
    CopyPoint(newshape,  shape, 35, 46);
    CopyPoint(newshape,  shape, 36, 39);
    InterPoint(newshape, shape, 37, .6667, 30, 40);
    newshape(37, IX) = shape(50, IX);
    CopyPoint(newshape,  shape, 38, 50);
    CopyPoint(newshape,  shape, 39, 58);
    CopyPoint(newshape,  shape, 40, 57);
    CopyPoint(newshape,  shape, 41, 56);
    CopyPoint(newshape,  shape, 42, 55);
    CopyPoint(newshape,  shape, 43, 54);
    CopyPoint(newshape,  shape, 44, 48);
    InterPoint(newshape, shape, 45, .3333, 30, 40);
    newshape(45, IX) = shape(48, IX);
    CopyPoint(newshape,  shape, 46, 51);
    CopyPoint(newshape,  shape, 47, 53);
    CopyPoint(newshape,  shape, 48, 59);
    CopyPoint(newshape,  shape, 49, 60);
    CopyPoint(newshape,  shape, 50, 61);
    CopyPoint(newshape,  shape, 51, 62);
    CopyPoint(newshape,  shape, 52, 63);
    CopyPoint(newshape,  shape, 53, 64);
    CopyPoint(newshape,  shape, 54, 65);
    CopyPoint(newshape,  shape, 55, 72);
    CopyPoint(newshape,  shape, 56, 73);
    CopyPoint(newshape,  shape, 57, 74);
    CopyPoint(newshape,  shape, 58, 75);
    CopyPoint(newshape,  shape, 59, 76);
    CopyPoint(newshape,  shape, 60, 69);
    CopyPoint(newshape,  shape, 61, 70);
    CopyPoint(newshape,  shape, 62, 71);
    CopyPoint(newshape,  shape, 63, 66);
    CopyPoint(newshape,  shape, 64, 67);
    CopyPoint(newshape,  shape, 65, 68);
    InterPoint(newshape, shape, 66, .5, 67, 70);
    CopyPoint(newshape,  shape, 67, 52);

#if MOD_A1 || MOD_A || MOD_A_EMU
    const double eyemouth = EyeMouthDist(shape);
    newshape(38, IY) += MAX(1, .05 * eyemouth); // move side of nose down
    newshape(44, IY) += MAX(1, .05 * eyemouth); // move side of nose down
    newshape(46, IY) += MAX(1, .02 * eyemouth); // move down, into nostril
    newshape(47, IY) += MAX(1, .02 * eyemouth); // move down, into nostril
#endif

    return newshape;
}

static Shape Shape77As68( // return an approximated XM2VTS 68 point shape
    const Shape& shape)   // in: Stasm 77 point shape
{
    CV_Assert(shape.rows == 77);

    Shape newshape(68, 2);

    CopyPoint(newshape,  shape,  0,  0);
    InterPoint(newshape, shape,  1, .6667, 1, 2);
    InterPoint(newshape, shape,  2, .5,    2, 3);
    CopyPoint(newshape,  shape,  3,  3);
    InterPoint(newshape, shape,  4, .3333, 3, 4);
    InterPoint(newshape, shape,  5, .6667, 4, 5);
    CopyPoint(newshape,  shape,  6,  5);
    CopyPoint(newshape,  shape,  7,  6);
    CopyPoint(newshape,  shape,  8,  7);
    InterPoint(newshape, shape,  9, .3333, 7, 8);
    InterPoint(newshape, shape, 10, .6667, 8, 9);
    CopyPoint(newshape,  shape, 11,  9);
    InterPoint(newshape, shape, 12, .5,    9, 10);
    InterPoint(newshape, shape, 13, .3333, 10, 11);
    CopyPoint(newshape,  shape, 14, 12);
    CopyPoint(newshape,  shape, 15, 25);
    CopyPoint(newshape,  shape, 16, 24);
    CopyPoint(newshape,  shape, 17, 23);
    CopyPoint(newshape,  shape, 18, 22);
    CopyPoint(newshape,  shape, 19, 27);
    CopyPoint(newshape,  shape, 20, 26);
    CopyPoint(newshape,  shape, 21, 18);
    CopyPoint(newshape,  shape, 22, 17);
    CopyPoint(newshape,  shape, 23, 16);
    CopyPoint(newshape,  shape, 24, 21);
    CopyPoint(newshape,  shape, 25, 20);
    CopyPoint(newshape,  shape, 26, 19);
    CopyPoint(newshape,  shape, 27, 34);
    CopyPoint(newshape,  shape, 28, 32);
    CopyPoint(newshape,  shape, 29, 30);
    CopyPoint(newshape,  shape, 30, 36);
    CopyPoint(newshape,  shape, 31, 38);
    CopyPoint(newshape,  shape, 32, 44);
    CopyPoint(newshape,  shape, 33, 42);
    CopyPoint(newshape,  shape, 34, 40);
    CopyPoint(newshape,  shape, 35, 46);
    CopyPoint(newshape,  shape, 36, 39);
    InterPoint(newshape, shape, 37, .6667, 30, 40);
    newshape(37, IX) = shape(50, IX);
    CopyPoint(newshape,  shape, 38, 50);
    CopyPoint(newshape,  shape, 39, 58);
    CopyPoint(newshape,  shape, 40, 57);
    CopyPoint(newshape,  shape, 41, 56);
    CopyPoint(newshape,  shape, 42, 55);
    CopyPoint(newshape,  shape, 43, 54);
    CopyPoint(newshape,  shape, 44, 48);
    InterPoint(newshape, shape, 45, .3333, 30, 40);
    newshape(45, IX) = shape(48, IX);
    CopyPoint(newshape,  shape, 46, 51);
    CopyPoint(newshape,  shape, 47, 53);
    CopyPoint(newshape,  shape, 48, 59);
    CopyPoint(newshape,  shape, 49, 60);
    CopyPoint(newshape,  shape, 50, 61);
    CopyPoint(newshape,  shape, 51, 62);
    CopyPoint(newshape,  shape, 52, 63);
    CopyPoint(newshape,  shape, 53, 64);
    CopyPoint(newshape,  shape, 54, 65);
    CopyPoint(newshape,  shape, 55, 72);
    CopyPoint(newshape,  shape, 56, 73);
    CopyPoint(newshape,  shape, 57, 74);
    CopyPoint(newshape,  shape, 58, 75);
    CopyPoint(newshape,  shape, 59, 76);
    CopyPoint(newshape,  shape, 60, 69);
    CopyPoint(newshape,  shape, 61, 70);
    CopyPoint(newshape,  shape, 62, 71);
    CopyPoint(newshape,  shape, 63, 66);
    CopyPoint(newshape,  shape, 64, 67);
    CopyPoint(newshape,  shape, 65, 68);
    InterPoint(newshape, shape, 66, .5, 67, 70);
    CopyPoint(newshape,  shape, 67, 52);

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
    Shape newshape = DimKeep(Shape77AsXm2vts68(shape), 76, 2);

    CopyPoint(newshape,  shape, 68, 33); // extra eyelid points
    CopyPoint(newshape,  shape, 69, 31);
    CopyPoint(newshape,  shape, 70, 37);
    CopyPoint(newshape,  shape, 71, 35);
    CopyPoint(newshape,  shape, 72, 43);
    CopyPoint(newshape,  shape, 73, 41);
    CopyPoint(newshape,  shape, 74, 47);
    CopyPoint(newshape,  shape, 75, 45);

    return newshape;
}

Shape ConvertShape(          // return shape with nlandmarks, return no rows if can't
    const Shape& shape,      // in
    int          nlandmarks) // in: shape.rows=no change, 17=shape17, anything else return no rows
                             //     if shape.rows=77, treat specially:
                             //     77=nochange, 76=muct76, 68=xm2vts, 22=ar, 20=bioid, 17=shape17
{
    Shape newshape;
    if (nlandmarks == shape.rows)
        newshape = shape.clone();
    else if (nlandmarks == 17)
        newshape = Shape17OrEmpty(shape);
    else if (shape.rows == 76) // MUCT 76
    {
        switch (nlandmarks)
        {
        case 68:
            newshape = DimKeep(shape.clone(), 68, 2);  // MUCT 68 and XM2VTS
            break;
        default:
            break;
        }
    }
    else if (shape.rows == 77) // MUCT 77 (Stasm version 4)
    {
        switch (nlandmarks)
        {
        case 20:
            newshape = Shape77As20(shape);  // BioID
            break;
        case 22:
            newshape = Shape77As22(shape);  // AR
            break;
        case 68:
            newshape = Shape77As68(shape);  // MUCT 68 and XM2VTS
            break;
        case 76:
            newshape = Shape77As76(shape); // MUCT 76 (Stasm version 3)
            break;
        default:
            break;
        }
    }
    CV_Assert(newshape.rows == nlandmarks || newshape.rows == 0);
    return newshape;
}

} // namespace stasm
