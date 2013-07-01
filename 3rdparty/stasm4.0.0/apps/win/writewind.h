// writewind.h: function to write the window as a BMP file
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_WRITEWIND_H
#define STASM_WRITEWIND_H

namespace stasm
{
void WriteWindowAsBmp(    // write image in window hwnd to disk
    HWND        hwnd,     // in: the window we want to write
    const char* path,     // in: path of image in window
    const char* appname); // in: prepended to new image name

} // namespace stasm
#endif // STASM_WRITEWIND_H
