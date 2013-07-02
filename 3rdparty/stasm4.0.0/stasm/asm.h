// asm.h: Active Shape Model class
//
// Freeing memory: In the current implementation, explicit memory release
// on Mod destruction is unneeded (because the MAT buffer pointers in Mod,
// ShapeMod, and DescMod point to constant data).  If you add a new descriptor
// model, you may need to add destructors.  Although even if that new class
// fails to release memory it's probably not serious, because the ASM
// model(s) are only destructed once, at program termination.
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_ASM_H
#define STASM_ASM_H

#include "stasmhash.h"

namespace stasm
{
static const int EYEMOUTH_DIST = 100;  // scale image to this before ASM search starts

static const double PYR_RATIO = 2;     // scale image by 2 at each pyramid level

static const int N_PYR_LEVS = 4;       // number of levs in image pyramid

static const int SHAPEMODEL_ITERS = 4; // shape model iterations per pyr level

//-----------------------------------------------------------------------------

class Mod // An ASM model for finding landmarks.
{         // If multiple model Stasm, will use a separate Mod for each yaw range.
public:
    Shape ModSearch_(                  // returns coords of the facial landmarks
        const Shape& startshape,       // in: startshape roughly positioned on face
        const Image& img,              // in: grayscale image (typically just ROI)
        const Shape* pinnedshape=NULL) // in: pinned landmarks, NULL if nothing pinned
    const;

    const Shape ConformShapeToMod_Pinned_( // wrapper around the func in ShapeMod
        const Shape& shape,                // in
        const Shape& pinnedshape)          // in
    const
    {
        VEC b(NSIZE(shapemod_.eigvals_), 1, 0.); // dummy variable for call below
        return shapemod_.ConformShapeToMod_Pinned_(b, shape, 0, pinnedshape);
    }

    // readonly access to some private vars
    ESTART       Estart_(void)    const { return estart_; }
    const char*  DataDir_(void)   const { return datadir_.c_str(); }
    const Shape  MeanShape_(void) const { return shapemod_.meanshape_; }

private:
    const vec_vec_BaseDescMod DescMods_(        // utility for Mod constructor
        const BaseDescMod** const descmods_arg, // in: descriptor models
        int                       ndescmods)    // in: sanity check
    {
        CV_Assert(ndescmods == stasm_NLANDMARKS * N_PYR_LEVS);

        vec_vec_BaseDescMod descmods(N_PYR_LEVS);

        for (int ilev = 0; ilev < N_PYR_LEVS; ilev++)
        {
            descmods[ilev].resize(stasm_NLANDMARKS);
            for (int i = 0; i < stasm_NLANDMARKS; i++)
                descmods[ilev][i] =
                    descmods_arg[ilev * stasm_NLANDMARKS + i];
        }
        return descmods;
    }

public:
    Mod(EYAW                eyaw,          // constructor
        ESTART              estart,
        string              datadir,
        Shape               meanshape,
        VEC                 eigvals,
        MAT                 eigvecs,
        int                 neigs,
        double              bmax,
        unsigned            hackbits,
        const BaseDescMod** descmods,
        int                 ndescmods)

        : eyaw_(eyaw),
          estart_(estart),
          datadir_(datadir),
          shapemod_(meanshape, eigvals, eigvecs, neigs, bmax, hackbits),
          descmods_(DescMods_(descmods, ndescmods))
    {
        CV_Assert(eyaw == EYAW_45 || eyaw == EYAW_22 || eyaw == EYAW00 ||
                  eyaw == EYAW22  || eyaw == EYAW45);
        CV_Assert(estart == ESTART_RECT_ONLY || estart == ESTART_EYES ||
                  estart == ESTART_EYE_AND_MOUTH);
    }

    virtual ~Mod() {}                      // destructor

private: // all data remains constant after Mod construction

    const EYAW     eyaw_;     // model is for this yaw range
                              // e.g. EYAW00 is frontal model

    const ESTART   estart_;   // use the mouth/eyes to position the start shape?

    const string   datadir_;  // directory of face detector files

    const ShapeMod shapemod_; // the shape model

    const vec_vec_BaseDescMod descmods_;
                              // descriptor mods, one for each point at each pyr lev
                              // index as [ilev][ipoint]

    void SuggestShape_(
        Shape&       shape,   // io: points will be moved to give best desc matches
        int          ilev,    // in: pyramid level (0 is full size)
        const Image& img,     // in: image scaled to this pyramid level
        const Shape& pinned,  // in: if no rows then no pinned landmarks, else
                              //     points except those equal to 0,0 are pinned
        const Hat &hat,
        StasmHash &hash)
    const;

    void LevSearch_(              // do an ASM search at one level in the image pyr
        Shape&       shape,       // io: the face shape for this pyramid level
        int          ilev,        // in: pyramid level (0 is full size)
        const Image& img,         // in: image scaled to this pyramid level
        const Shape& pinnedshape) // in: if no rows then no pinned landmarks, else
                                  //     points except those equal to 0,0 are pinned
    const;

    DISALLOW_COPY_AND_ASSIGN(Mod);

}; // end class Mod

typedef vector<const Mod*> vec_Mod; // vector of ASM models, one for each yaw range

//-----------------------------------------------------------------------------

static inline double GetPyrScale(  // return 1 for pyr lev 0, .5 for pyr lev 1, etc.
    int ilev)                      // in: pyramid level (0 is full size)
{
    return 1 / pow(PYR_RATIO, ilev);
}

// TODO This definition doesn't belong here.
int EyawAsModIndex(       // note: returns a negative index for left facing yaws
    EYAW           eyaw,  // in
    const vec_Mod& mods); // in: a vector of models, one for each yaw range

} // namespace stasm
#endif // STASM_ASM_H
