// hatdesc.h: Model for HAT descriptors
//            This does a search using the descriptors from hat.cpp.
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_HATPATCH_H
#define STASM_HATPATCH_H

#include "stasmhash.h"

namespace stasm
{
static const int HAT_MAX_OFFSET = 4;   // search grid +-4 pixs from current posn

static const int HAT_SEARCH_RESOL = 2; // search resolution, search every 2nd pixel

// Following params must match those used for generating the HAT
// descriptors used to generate the HAT models during training.

static const int HAT_PATCH_WIDTH = 9*2+1;
                                       // HAT patch is 19 x 19 at pyr lev 0

static const int HAT_PATCH_WIDTH_ADJ = -6;
                                       // grid gets smaller for smaller pyr levs

static const int HAT_START_LEV = 2;    // HAT descriptors are for pyr levs 0...2
                                       // so no need for Hat::Init_ at pyr lev 3

// define HatFit: a pointer to a func for measuring fit of HAT descriptor
typedef double(*HatFit)(const double* const);

extern Hat InitHatLevData( // init the global HAT data needed for this pyr level
    const Image& img,       // in
    int          ilev);     // in: pyramid level, 0 is full size

extern void HatDescSearch(  // search in a grid around the current landmark
    double&      x,         // io: (in: old posn of landmark, out: new posn)
    double&      y,         // io
    const HatFit hatfit,   // in: func to estimate descriptor match
    const Hat &hat,
    StasmHash &hash);

class HatDescMod: public BaseDescMod
{
public:
    virtual void DescSearch_(double& x, double& y,       // io
                             const Image&, const Shape&, // in
                             int, int, const Hat &hat, StasmHash &hash) const             // in
    {
        HatDescSearch(x, y, hatfit_, hat, hash);
    }

    HatDescMod(const HatFit hatfit) // constructor
        : hatfit_(hatfit)
    {
    }

private:
    HatFit const hatfit_; // func to estimate HAT descriptor match

    DISALLOW_COPY_AND_ASSIGN(HatDescMod);

}; // end class HatDescMod

} // namespace stasm
#endif // STASM_HATPATCH_H
