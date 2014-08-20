// initasm.h: initialize the ASM model
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_INITASM_H
#define STASM_INITASM_H

namespace stasm
{
void InitMods(
    vec_Mod&    mods,     // out: ASM model (only one model in this version of Stasm)
    const char* datadir); // in: directory of face detector files

} // namespace stasm
#endif // STASM_INITASM_H
