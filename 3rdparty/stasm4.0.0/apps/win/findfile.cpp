// findfile.cpp: Routines for scrolling through all files in a directory.
//               Current version is Microsoft specific.
//
// Typical usage:
//
//      InitFindFile("c:/images", ".jpg;.bmp;.png");
//      while (1)
//      {
//          char path[MAX_PATH];
//
//          NextFile(path, 1); // sets path to next image filename in c:/images
//                             // the "1" means "next file forwards"
//
//          process path, possibly break from loop...
//      }

#include <windows.h>
#include <commctrl.h>
#include <io.h>
#include "stasm.h"
#include "findfile.h"

namespace stasm
{
#pragma warning(disable: 4996)  // 'strtok' This function may be unsafe

static vector<string> paths_g;  // circular buffer of file paths

static int            ipath_g;  // index into circular buffer

//-----------------------------------------------------------------------------

static bool InStrings( // true if string is in semicolon-separated list of strings
    const char* string,            // in
    const char* strings,           // in: semicolon-separated list of strings
    bool        ignore_case=false) // in
{
    // following def needed because strtok replaces ; with a 0
    char strings1[SLEN]; STRCPY(strings1, strings);
    const char* token = strtok(strings1, ";");
    while (token)
    {
        if (ignore_case)
        {
            if (_stricmp(string, token) == 0)
                return true;
        }
        else
        {
            if (strcmp(string, token) == 0)
                return true;
        }
        token = strtok(NULL, ";");
    }
    return false;
}

static bool MatchingExt( // true if FindData matches one of given extensions
    const struct _finddata_t& finddata,
    const char*               exts)
{
    char ext[_MAX_EXT]; splitpath(finddata.name, NULL, NULL, NULL, ext);
    return !(finddata.attrib & (_A_HIDDEN|_A_SUBDIR|_A_SYSTEM)) &&
           InStrings(ext, exts, true);
}

void InitFindFile(    // must be called before NextFile
    const char* path, // in: specify directory (only the dir of this path is used)
    const char* exts) // in: semi-colon separated list of valid extensions
{
    char first[SLEN]; STRCPY(first, path);
    ConvertBackslashesToForwardAndStripFinalSlash(first);

    // use first to form a wildcard (C:/dir/image.jpg becomes C:/dir/*.*)

    char drive[_MAX_DRIVE], dir[_MAX_DIR];
    splitpath(first, drive, dir, NULL, NULL);
    char wildcard[SLEN]; makepath(wildcard, drive, dir, "*", "*");

    // put all image paths that match wildcard into paths_g

    paths_g.resize(0);
    struct _finddata_t finddata;
    intptr_t hfile;
    hfile = _findfirst(wildcard, &finddata);
    if (hfile)
    {
        char drivedir[SLEN]; makepath(drivedir, drive, dir, NULL, NULL);
        ConvertBackslashesToForwardAndStripFinalSlash(drivedir);
        if (MatchingExt(finddata, exts))
            paths_g.push_back(ssprintf("%s/%s", drivedir, finddata.name));
        while (_findnext(hfile, &finddata) == 0)
        {
            if (MatchingExt(finddata, exts))
                paths_g.push_back(ssprintf("%s/%s", drivedir, finddata.name));
        }
    }
    // set ipath_g to index the entry in paths_g that matches first
    for (ipath_g = 0;
         ipath_g < NSIZE(paths_g) && _stricmp(paths_g[ipath_g].c_str(), first);
         ipath_g++)
        ;
    if (ipath_g >= NSIZE(paths_g)) // not found?
        ipath_g = 0;               // must have been deleted indep of this app
}

static int NextFile1( // return new file index
    char* path,       // out: path of next image in directory
    int   offset)     // in: typically 1, -1, FINDFILE_FIRST, or FINDFILE_LAST
{
    const int nfiles = NSIZE(paths_g);
    if (nfiles == 0)
    {
        // not sure what is best here
        ipath_g = 0;
        path = "";
        return 0;
    }
    int ipath = ipath_g;
    if (offset == FINDFILE_FIRST)
        ipath = 0;
    else if (offset == FINDFILE_LAST)
        ipath = nfiles - 1;
    else
    {
        if (offset >= nfiles)
            offset = 1;
        else if (offset <= -nfiles)
            offset = -1;
        ipath += offset;
        if (ipath < 0)
            ipath += nfiles;
        else if (ipath >= nfiles)
            ipath -= nfiles;
    }
    CV_Assert(ipath >= 0 && ipath < nfiles);
    strcpy_s(path, SLEN, paths_g[ipath].c_str());
    return ipath;
}

bool NextFile(    // updates ipath_g,  return true if more than one image
    char* path,   // out: path of next image in directory
    int   offset) // in: typically 1, -1, FINDFILE_FIRST, or FINDFILE_LAST
{
    ipath_g = NextFile1(path, offset);
    return NSIZE(paths_g) > 0;
}

void NextFilePeek( // does not update ipath_g
    char* path,    // out: path of next image in directory
    int   offset)  // in: typically 1, -1, FINDFILE_FIRST, or FINDFILE_LAST
{
    NextFile1(path, offset);
}

} // namespace stasm
