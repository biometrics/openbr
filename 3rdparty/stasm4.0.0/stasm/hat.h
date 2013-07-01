// hat.h: Histogram Array Transform descriptors
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_HAT_H
#define STASM_HAT_H

namespace stasm
{
class Hat
{
public:
    void Init_(                   // init the HAT internal grad mat and indices
        const Image& img,         // in: image ROI scaled to the current pyr lev
        const int    patchwidth); // in: patch will be patchwidth x patchwidth pixs

    VEC Desc_(                    // return HAT descriptor, Init_ must be called first
        const double x,           // in: x coord of center of patch (may be off image)
        const double y)           // in: y coord of center of patch (may be off image)
    const;

    Hat() {}                      // constructor

private:
    // All these private variables are initialized by Hat::Init_.  They must
    // be initialized if the image changes or if the patch width changes.
    // (In a Stasm context, that means they must be initialized once per
    // pyramid level.  Also, for safe use of OpenMP in SuggestShape_,
    // they must not change unless the pyramid level changes.)

    int        patchwidth_;       // image patch is patchwidth x patchwidth pixels

    MAT        magmat_;           // grad mag of the current image (face ROI)
    MAT        orientmat_;        // grad orient of the current image (face ROI)

    vec_int    row_indices_;      // histogram indices: these map a patch row,col
    vec_double row_fracs_;        // to the corresponding histogram grid row,col
    vec_int    col_indices_;
    vec_double col_fracs_;

    vec_double pixelweights_;     // weight pixel by closeness to center of patch

    //DISALLOW_COPY_AND_ASSIGN(Hat);

}; // end class Hat

} // namespace stasm
#endif // STASM_HAT_H
