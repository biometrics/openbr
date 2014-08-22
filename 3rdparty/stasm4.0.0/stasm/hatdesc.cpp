// hatdesc.cpp: Model for HAT descriptors.
//              This does a grid search using the descriptors defined hat.cpp.
//              It also caches the descriptors for fast reuse.
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"
#include "stasmhash.h"

#define CACHE 1   // define to 0 if your compiler doesn't support hash_map
                  // Stasm runs faster if 1

namespace stasm
{
// hat_g is global because we initialize the HAT internal data
// (grads and orients etc.) once for the entire pyramid level.
// Initialized in InitHatLevData.

//static const Hat &hat_g;

//-----------------------------------------------------------------------------

#if CACHE

// For speed, we cache the HAT descriptors, so we have the descriptor at
// hand if we revisit an xy position in the image which is very common in ASMs.
// (Note: an implementation with cache_g as a vector<vector VEC>> was slower.)

//static hash_map<unsigned, VEC> cache_g; // cached descriptors
//static const bool TRACE_CACHE = 0;      // for checking cache hit rate
//static int ncalls_g, nhits_g;           // only used if TRACE_CACHE

static unsigned Key(int x, int y) // pack x,y into 32 bits for cache key
{
    return ((y & 0xffff) << 16) | (x & 0xffff);
}

static double GetHatFit( // args same as non CACHE version, see below
    int          x,      // in
    int          y,      // in
    const HatFit hatfit, // in
    const Hat &hat,
    StasmHash &hash)
{
    const double* descbuf = NULL;       // the HAT descriptor

    const unsigned key(Key(x, y));

    QHash<unsigned, VEC>::const_iterator it = hash.find(key);
    if (it != hash.end()) {
        descbuf = Buf(it.value());
    } else {
        const VEC desc(hat.Desc_(x, y));
        hash[key] = desc;            // remember descriptor for possible re-use
        descbuf = Buf(desc);
    }
    return hatfit(descbuf);
}

#else // not CACHE

// Get the HAT descriptor at the given ipoint and x,y coords, and return
// how well the descriptor matches the model.  High fit means good match.

static double GetHatFit(
    int          x,      // in: image x coord (may be off image)
    int          y,      // in: image y coord (may be off image)
    const HatFit hatfit) // in: func to estimate descriptor match
{
    return hatfit(Buf(hat_g.Desc_(x, y)));
}

#endif // not CACHE

static int round2(double x) // return closest int to x that is divisible by 2
{
    return 2 * cvRound(x / 2);
}

static int PatchWidth( // patchwidth at the given pyramid level
    int ilev)          // in: pyramid level (0 is full size)
{
    return HAT_PATCH_WIDTH + round2(ilev * HAT_PATCH_WIDTH_ADJ);
}

Hat InitHatLevData(   // init the global HAT data needed for this pyr level
    const Image& img,  // in
    int          ilev) // in
{
    Hat hat;

    if (ilev <= HAT_START_LEV) // we use HATs only at upper pyr levs
    {
        hat.Init_(img, PatchWidth(ilev));
    }

    return hat;
}

// Note 1: The image is not passed directly to this function.  Instead this
// function accesses the image gradient magnitude and orientation stored in
// fields of the global variable hat_g and previously initialized by the
// call to InitHatLevData.
//
// Note 2: If OpenMP is enabled, multiple instances of this function will be
// called concurrently (each call will have a different value of x and y). Thus
// this function and its callees do not modify any data that is not on the stack.

void HatDescSearch(      // search in a grid around the current landmark
    double&      x,      // io: (in: old position of landmark, out: new position)
    double&      y,      // io:
    const HatFit hatfit, // in: func to estimate descriptor match
    const Hat &hat,
    StasmHash &hash)
{
    // If HAT_SEARCH_RESOL is 2, force x,y positions to be divisible
    // by 2 to increase cache hit rate. This increases the mean hit rate
    // from about 67% to 88% and barely affects landmark accuracy.
    int ix = HAT_SEARCH_RESOL == 2? round2(x): cvRound(x);
    int iy = HAT_SEARCH_RESOL == 2? round2(y): cvRound(y);

    double fit_best = -FLT_MAX;

    int xoffset_best = 0, yoffset_best = 0; // in pixels

    for (int yoffset = -HAT_MAX_OFFSET;
             yoffset <= HAT_MAX_OFFSET;
             yoffset += HAT_SEARCH_RESOL)
    {
        for (int xoffset = -HAT_MAX_OFFSET;
                 xoffset <= HAT_MAX_OFFSET;
                 xoffset += HAT_SEARCH_RESOL)
        {
            const double fit = GetHatFit(ix + xoffset, iy + yoffset, hatfit, hat, hash);
            if (fit > fit_best)
            {
                fit_best = fit;
                xoffset_best = xoffset;
                yoffset_best = yoffset;
            }
        }
    }
    x += xoffset_best;
    y += yoffset_best;
}

} // namespace stasm
