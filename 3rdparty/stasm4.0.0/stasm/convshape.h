// convshape.cpp: convert a stasm 77 point shape to other formats
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_CONVSHAPE_H
#define STASM_CONVSHAPE_H

namespace stasm
{
Shape ConvertShape(           // return shape with nlandmarks, return no rows if can't
    const Shape& shape,       // in
    int          nlandmarks); // in: 77=nochange, 76=stasm3, 68=xm2vts, 22=ar, 20=bioid, 17=me17

} // namespace stasm
#endif // STASM_CONVSHAPE_H
