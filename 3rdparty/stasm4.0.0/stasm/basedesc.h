// basedesc.h: descriptor model base class
//
// A "descriptor model" tells you how to use the given
// descriptor to suggest the best position of a landscape.
//
// Stasm currently uses two types of descriptors, and thus two descriptor
// model classes: ClassicDescMod and HatDescMod.  Each of these classes
// provide a descriptor matching function, DescSearch_, which searches around
// the current position of a landmark, looking for the best new position.
//
// About BaseDescMod: We need a vector of descriptor models to specify how
// to search by matching against a descriptor at each landmark.  Some
// landmarks use ClassicDescMods; other use HatDescMods.  We thus need a
// vector of heterogeneous objects.  But C++ doesn't support vectors of
// heterogeneous objects.  So instead we use a vector of pointers to
// BaseDescMod, with the actual descriptor model classes (ClassicDescMod
// and HatDescMod) deriving from BaseDescMod.
//
// Memory release: Explicit destructors unneeded, see note in header of asm.h.
//
// DescSearch_ and concurrency: If OpenMP is enabled, DescSearch_ will be
// called concurrently for multiple points.  (Each call will have a
// different value of x and y.)  Thus for the OpenMP code to work
// correctly, DescSearch_ and its callees must not modify any variables that
// are not on the stack unless the variable is protected by a critical region.
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_BASEDESC_H
#define STASM_BASEDESC_H

#include "hat.h"
#include "stasmhash.h"

namespace stasm
{
class BaseDescMod // abstract base class for all descriptor models
{
public:
    virtual void DescSearch_( // search in area around the current point
        double&      x,       // io: (in: old posn of landmark, out: new posn)
        double&      y,       // io
        const Image& img,     // in: image scaled to this pyramid level
        const Shape& shape,   // in: current position of the landmarks
        int          ilev,    // in: pyramid level (0 is full size)
        int          ipoint,  // in: index of the current landmark
        const Hat    &hat,
        StasmHash    &hash)
    const = 0;

    virtual ~BaseDescMod() {} // destructor
};

// vec_vec_BaseDescMod contains the descriptor models, one pointer
// for each landmark at each pyramid level, index as [ilev][ipoint]

typedef vector<vector<const BaseDescMod*> > vec_vec_BaseDescMod;

} // namespace stasm
#endif // STASM_BASEDESC_H
