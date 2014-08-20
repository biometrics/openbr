// shapehacks.h:
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_SHAPEHACKS_H
#define STASM_SHAPEHACKS_H

namespace stasm
{
static const unsigned SHAPEHACKS_DEFAULT          = 0x01;
static const unsigned SHAPEHACKS_SHIFT_TEMPLE_OUT = 0x10; // for frontal models

void ApplyShapeModelHacks( // adjust shape by applying various hacks
    Shape&   shape,        // io: features possibly adjusted
    unsigned hackbits);    // in: which hacks to apply, see above constants

} // namespace stasm
#endif // STASM_SHAPEHACKS_H
