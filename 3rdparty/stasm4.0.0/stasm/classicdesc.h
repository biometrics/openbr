// classicdesc.h: model for classic ASM descriptors
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_CLASSICDESC_H
#define STASM_CLASSICDESC_H

#include "stasmhash.h"

namespace stasm
{
static const int CLASSIC_MAX_OFFSET = 2;   // search +-2 pixels along the whisker
static const int CLASSIC_SEARCH_RESOL = 2; // search resolution, every 2nd pix

extern void ClassicDescSearch( // search along whisker for best profile match
    double&      x,        // io: (in: current posn of the point, out: new posn)
    double&      y,        // io:
    const Image& img,      // in: the image scaled to this pyramid level
    const Shape& inshape,  // in: current posn of landmarks (for whisker directions)
    int          ipoint,   // in: index of the current landmark
    const MAT&   meanprof, // in: mean of the training profiles for this point
    const MAT&   covi);    // in: inverse of the covar of the training profiles

class ClassicDescMod: public BaseDescMod
{
public:
    virtual void DescSearch_(double& x, double& y,                 // io
                             const Image& img, const Shape& shape, // in
                             int, int ipoint, const Hat &hat, StasmHash &hash) const                // in
    {
        (void) hat;
        (void) hash;

        ClassicDescSearch(x, y, img, shape, ipoint, meanprof_, covi_);
    }

    ClassicDescMod(                          // constructor
        int                 profwidth,
        const double* const meanprof_data,
        const double* const covi_data)

        : meanprof_(ArrayAsMat(1, profwidth, meanprof_data)),
          covi_(ArrayAsMat(profwidth, profwidth, covi_data))
    {
    }

private:
    const MAT meanprof_; // mean of the training profiles for this point
    const MAT covi_;     // inverse of the covariance of the training profiles

    DISALLOW_COPY_AND_ASSIGN(ClassicDescMod);

}; // end class ClassicDescMod


} // namespace stasm
#endif // STASM_CLASSICDESC_H
