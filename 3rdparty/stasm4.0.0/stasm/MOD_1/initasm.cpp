// initasm.cpp: initialize the ASM model
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "../stasm.h"
#include "yaw00.h"

namespace stasm
{
void InitMods(           // initialize ASM model
    vec_Mod&    mods,    // out: ASM model (only one model in this version of Stasm)
    const char* datadir) // in: directory of face detector files
{
    if (mods.empty())    // models not yet initialized?
    {
        mods.resize(1);  // 1 model

        static const Mod mod_yaw00(
            EYAW00,
            ESTART_EYES, // ignore detected mouth for best startshape on frontal faces
            datadir,
            yaw00_meanshape,
            yaw00_eigvals,
            yaw00_eigvecs,
            20,  // neigs (value from empirical testing)
            1.5, // bmax  (value from empirical testing)
            SHAPEHACKS_DEFAULT | SHAPEHACKS_SHIFT_TEMPLE_OUT,
            YAW00_DESCMODS, // defined in yaw00.h
            NELEMS(YAW00_DESCMODS));

        mods[0] = &mod_yaw00;
    }
}

} // namespace stasm
