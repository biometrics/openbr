// shapemod.h: the ASM shape model
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_SHAPEMODEL_H
#define STASM_SHAPEMODEL_H

namespace stasm
{
class ShapeMod
{
public:
    const Shape ConformShapeToMod_( // return shape conformed to shape mod
        VEC&         b,             // io: eigvec weights
        const Shape& shape,         // in: shape suggested by the descriptor mods
        int          ilev)          // in: pyramid level (0 is full size)
    const;

    const Shape ConformShapeToMod_Pinned_( // like above but allow pinned points
        VEC&         b,             // io: eigvec weights
        const Shape& shape,         // in: shape suggested by the descriptor mods
        int          ilev,          // in: pyramid level (0 is full size)
        const Shape& pinnedshape)   // in: pinned landmarks
    const;

    ShapeMod(                       // constructor
        const Shape&   meanshape,
        const VEC&     eigvals,
        const MAT&     eigvecs,
        const int      neigs,
        const double   bmax,
        const unsigned hackbits)

        : meanshape_(meanshape),
          eigvals_(DimKeep(eigvals, neigs, 1)),
          eigvecs_(DimKeep(eigvecs, eigvecs.rows, neigs)), // retain neigs cols
          // take inverse of eigvecs (by taking transpose) and retain neigs rows
          eigvecsi_(DimKeep(eigvecs.t(), neigs, eigvecs.cols)),
          bmax_(bmax),
          hackbits_(hackbits)
    {
        CV_Assert(meanshape.rows == stasm_NLANDMARKS);
        CV_Assert(meanshape.cols == 2);
        CV_Assert(NSIZE(eigvals) == 2 * stasm_NLANDMARKS);
        CV_Assert(eigvecs.rows   == 2 * stasm_NLANDMARKS);
        CV_Assert(eigvecs.cols   == 2 * stasm_NLANDMARKS);
        CV_Assert(neigs > 0 && neigs <= 2 * stasm_NLANDMARKS);
        CV_Assert(bmax > 0 && bmax < 10);
        CV_Assert((hackbits & ~(SHAPEHACKS_DEFAULT|SHAPEHACKS_SHIFT_TEMPLE_OUT)) == 0);
    }

    // all data remains constant after ShapeMod construction

    const Shape    meanshape_; // mean shape aligned to face det frame
    const VEC      eigvals_;   // neigs x 1 vector
    const MAT      eigvecs_;   // 2n x neigs matrix where n is nbr of landmarks
    const MAT      eigvecsi_;  // neigs x 2n matrix, inverse of eigvecs_
    const double   bmax_;      // eigvec weight limit, for LimitB()
    const unsigned hackbits_;  // allowable shape model hacks (e.g. SHAPEHACKS_DEFAULT)

private:
    DISALLOW_COPY_AND_ASSIGN(ShapeMod);

}; // end class ShapeMod

} // namespace stasm
#endif // STASM_SHAPEMODEL_H
