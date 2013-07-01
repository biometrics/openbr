// findfile.h: Routines for scrolling through all files in a dir.
//             Current version is Microsoft specific.
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_FINDFILE_H
#define STASM_FINDFILE_H

namespace stasm
{
static const int FINDFILE_FIRST = INT_MIN; // NextFile return first file in dir
static const int FINDFILE_LAST  = INT_MAX; // NextFile return last file in dir

void InitFindFile(     // must be called before NextFile
    const char* path,  // in: only the directory of this path is used
    const char* exts); // in: semi-colon separated list of valid extensions

bool NextFile(         // get filename at current index, bump index for next time
                       // return true if more than one file in dir
    char* path,        // out: path of next image in directory
    int   offset);     // in: typically 1, -1, FINDFILE_FIRST, or FINDFILE_LAST

void NextFilePeek(     // like NextFile but doesn't change internal file index
    char* path,        // out: path of next image in directory
    int   offset);     // in: typically 1, -1, FINDFILE_FIRST, or FINDFILE_LAST

} // namespace stasm
#endif // STASM_FINDFILE_H
