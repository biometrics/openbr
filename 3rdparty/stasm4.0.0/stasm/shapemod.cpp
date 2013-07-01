// shapemod.cpp: the ASM shape model
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"

namespace stasm
{
static const int SHAPEHACK_MINPYRLEV = 2; // allow hacks only at coarse pyr levs

// Limit the values of b to make sure the generated shape is plausible.
// That is, clip each b[i] to bmax * sqrt(lambda_i).
// "b" is the name used for the eigenvector weights in Cootes' papers.

static void LimitB(
    VEC&       b,       // io: eigvec weights
    const VEC& eigvals, // in
    double     bmax)    // in
{
    for (int i = 0; i < NSIZE(eigvals); i++)
    {
        const double limit = bmax * sqrt(eigvals(i));
        b(i) = Clamp(b(i), -limit, limit);
    }
}

// This implements Section 4.8 of CootesTaylor 2004
// www.isbe.man.ac.uk/~bim/Mods/app_models.pdf.
// Except that we don't implement tangent spaces.  And we don't iterate the
// shape model until convergence.  Instead we use the b from the previous
// iteration of the ASM, which gives as good landmark fit results, empirically.

static Shape ConformShapeToMod( // Return a copy of inshape conformed to the model
    VEC&         b,             // io: eigvec weights
    const Shape& inshape,       // in: the current position of the landmarks
    const Shape& meanshape,     // in: n x 2
    const VEC&   eigvals,       // in: neigs x 1
    const MAT&   eigvecs,       // in: 2n x neigs
    const MAT&   eigvecsi,      // in: neigs x 2n, inverse of eigvecs
    const double bmax,          // in: for LimitB
    const VEC&   pointweights)  // in: contribution of each point to the pose
{
    Shape shape(inshape.clone());

    // estimate the pose which transforms the shape into the model space
    // (use the b from previous iterations of the ASM)

    MAT modelshape(AsColVec(meanshape) + eigvecs * b);
    modelshape = DimKeep(modelshape, shape.rows, 2); // redim back to 2 columns
    const MAT pose(AlignmentMat(modelshape, shape, Buf(pointweights)));

    // transform the shape into the model space

    modelshape = AlignShape(shape, pose.inv(cv::DECOMP_LU));

    // update shape model params b to match modelshape, then limit b

    b = eigvecsi * AsColVec(modelshape - meanshape);
    LimitB(b, eigvals, bmax);

    // generate conformedshape from the model using the limited b
    // (we generate as a column vec, then redim back to 2 columns)

    const Shape conformedshape(DimKeep(eigvecs * b, shape.rows, 2));

    // back to image space

    return AlignShape(meanshape + conformedshape, pose);
}

static VEC PointWeights(void) // return point weights from LANDMARK_INFO_TAB
{
    CV_DbgAssert(NELEMS(LANDMARK_INFO_TAB) == stasm_NLANDMARKS);

    VEC pointweights(stasm_NLANDMARKS, 1);

    for (int i = 0; i < stasm_NLANDMARKS; i++)
        pointweights(i) = LANDMARK_INFO_TAB[i].weight;

    return pointweights;
}

// wrapper around ConformShapeToMod above

const Shape ShapeMod::ConformShapeToMod_( // return shape conformed to shape model
    VEC&         b,     // io: eigvec weights from previous iters of ASM
    const Shape& shape, // in: shape suggested by the descriptor models
    int          ilev)  // in: pyramid level (0 is full size)
const
{
    // static for efficiency (init once)
    static const VEC pointweights(PointWeights());

    Shape newshape = ConformShapeToMod(b,
                        shape, meanshape_ * GetPyrScale(ilev),
                        eigvals_ / pow(SQ(PYR_RATIO), ilev), eigvecs_, eigvecsi_,
                        bmax_, pointweights);

    newshape = JitterPointsAt00(newshape); // jitter points at 0,0 if any

    if (ilev >= SHAPEHACK_MINPYRLEV) // allow shape hacks only at coarse pyr levs
        ApplyShapeModelHacks(newshape, hackbits_);

    return newshape;
}

// Like ConformShapeToMod_ but with pinned landmarks.  Conform the given shape to
// the ASM model, but keeping points in pinnedshape at their original position.

const Shape ShapeMod::ConformShapeToMod_Pinned_(
    VEC&         b,           // io: eigvec weights from previous iters of ASM
    const Shape& shape,       // in: shape suggested by the descriptor models
    int          ilev,        // in: pyramid level (0 is full size)
    const Shape& pinnedshape) // in: pinned landmarks
const
{
    static const double MAX_DIST = 0.5;
    static const int    MAX_ITERS = 50;

    Shape outshape(shape.clone());
    double dist = FLT_MAX;
    for (int iter = 0; dist > MAX_DIST && iter < MAX_ITERS; iter++)
    {
        outshape = ConformShapeToMod_(b, outshape, ilev);
        dist = ForcePinnedPoints(outshape, pinnedshape);
    }
    return outshape;
}

} // namespace stasm
