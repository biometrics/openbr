// fitmeas.h: calculate fitness measures
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_FITMEAS_H
#define STASM_FITMEAS_H

namespace stasm
{
static const double NOFIT = .99999; // indicate that the fit is not avail

double MeanFitOverInterEye( // mean landmark distance divided by intereye
    int&         iworst,    // out: index of point with worse fit
    const Shape& shape,     // in
    const Shape& refshape); // in: shape will be converted to same nbr of points as refshape

double Me17(                // me17 fitness measure
    const Shape& shape,     // in
    const Shape& refshape); // in

} // namespace stasm
#endif // STASM_FITMEAS_H
