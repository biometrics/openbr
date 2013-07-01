// fm29.h: calculate the FM29 measure of fitness
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_FM29_H
#define STASM_FM29_H

namespace stasm
{
void Fm29(
    double&      fm29,      // out: FM29 measure of fitness
    int&         iworst,    // out: index of point with worse fit
    const Shape& shape,     // in
    const Shape& refshape); // in

} // namespace stasm
#endif // STASM_FM29_H
