// writewind.cpp: function to write the window as a BMP file

#include <windows.h>
#include <commctrl.h>
#include <winuser.h>
#include <vfw.h>
#include <process.h>
#include <io.h>
#include <direct.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "stasm.h"
#include "usermsg.h"
#include "writewind.h"

namespace stasm
{
// Get a bmp path from the user.  Will only accept names ending in .bmp.

static void BmpPath(
    HWND  hwnd,      // in
    char* path)      // io: in:  default path, "" for none
                     //     out: path got from user, "" if none
{
    static char lpstrFile[SLEN];
    if (path[0] != 0)
        STRCPY(lpstrFile, path);
    OPENFILENAME ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = hwnd;
    ofn.lpstrFile = lpstrFile;
    ofn.nMaxFile = SLEN;
    ofn.lpstrFilter = "BMP\0*.bmp\0All\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.lpstrDefExt = "bmp"; // append bmp to path if no user extension
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_HIDEREADONLY |
                OFN_OVERWRITEPROMPT | OFN_NOREADONLYRETURN |
                OFN_EXTENSIONDIFFERENT;
    path[0] = 0;
    if (!GetSaveFileName(&ofn)) // displays open file dialog box
    {
        DWORD nErr = CommDlgExtendedError();
        if (nErr)
            UserMsg(hwnd, "Cannot create %s", path);
    }
    else if (ofn.Flags & OFN_EXTENSIONDIFFERENT)
        UserMsg(hwnd,
                "Cannot create %s\nIt has a bad extension (expected .bmp)",
                ofn.lpstrFile);
    else
        strcpy_s(path, SLEN, ofn.lpstrFile);
}

static void Fwrite(    // like fwrite but displays a message if write fails
    HWND        hwnd,  // in
    const void* buf,   // in
    size_t      size,  // in
    size_t      count, // in
    FILE*       file,  // in
    const char* path)  // in: used only for error messages
{
    if (fwrite(buf, size, count, file) != count)
        UserMsg(hwnd, "Cannot write %s", path);
}

// TODO Would be simpler to use opencv::imwrite here?

static void WriteWindowAsBmp1( // write BMP file of the given window
    HWND        hwnd,          // in: the window we want the image of
    const char* path)          // in: the file path
{
    RECT rect; GetClientRect(hwnd, &rect); // dimensions of hwnd
    HDC hdc = GetDC(hwnd); CV_Assert(hdc);
    HDC hdcMem = CreateCompatibleDC(hdc); CV_Assert(hdcMem);
    HBITMAP hBitmap = CreateCompatibleBitmap(hdc,
                              rect.right  - rect.left,
                              rect.bottom - rect.top);
    CV_Assert(SelectObject(hdcMem, hBitmap));
    CV_Assert(BitBlt(hdcMem, 0, 0, rect.right-rect.left, rect.bottom-rect.top, // dest
                     hdc, 0, 0, SRCCOPY));                                     // src
    BITMAP bmp; GetObject(hBitmap, sizeof(BITMAP), &bmp);

    BITMAPINFOHEADER bmiHeader;
    bmiHeader.biSize          = sizeof(bmiHeader);
    bmiHeader.biWidth         = bmp.bmWidth;
    bmiHeader.biHeight        = bmp.bmHeight;
    bmiHeader.biPlanes        = 1;
    bmiHeader.biBitCount      = 32;
    bmiHeader.biCompression   = BI_RGB; // no compression
    bmiHeader.biSizeImage     = 0;
    bmiHeader.biXPelsPerMeter = 0;
    bmiHeader.biYPelsPerMeter = 0;
    bmiHeader.biClrUsed       = 0;
    bmiHeader.biClrImportant  = 0;

    int bmp_size = ((bmp.bmWidth * bmiHeader.biBitCount + 31) / 32) * 4 * bmp.bmHeight;
    HANDLE hDIB = GlobalAlloc(GHND, bmp_size);
    char* bitmap = (char*)GlobalLock(hDIB);
    GetDIBits(hdc, hBitmap,
              0,(UINT)bmp.bmHeight, bitmap,
             (BITMAPINFO*)&bmiHeader, DIB_RGB_COLORS);

    BITMAPFILEHEADER bmfHeader;
    bmfHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    bmfHeader.bfSize = bmp_size + sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    bmfHeader.bfType = 0x4D42; // magic number for .bmp files

    FILE* file;
    if (fopen_s(&file, path, "wb") != 0)
        UserMsg(hwnd, "Cannot write %s", path);
    else
    {
        Fwrite(hwnd, (LPSTR)&bmfHeader, 1, sizeof(BITMAPFILEHEADER), file, path);
        Fwrite(hwnd, (LPSTR)&bmiHeader, 1, sizeof(BITMAPINFOHEADER), file, path);
        Fwrite(hwnd, (LPSTR)bitmap, 1, bmp_size, file, path);
        fclose(file);
    }
    GlobalUnlock(hDIB);
    GlobalFree(hDIB);
    DeleteObject(hBitmap);
    DeleteObject(hdcMem);
    ReleaseDC(hwnd, hdc);
}

void WriteWindowAsBmp(   // write image in window hwnd to disk
    HWND        hwnd,    // in: the window we want to write
    const char* path,    // in: path of image in window
    const char* appname) // in: prepended to new image name
{
    char appname1[SLEN]; STRCPY(appname1, appname); ToLowerCase(appname1);
    char writepath[SLEN];
    sprintf_s(writepath, SLEN, "%s_%s.bmp", appname1, Base(path));
    BmpPath(hwnd, writepath);               // get path from the user
    if (writepath[0])                       // valid path?
    {
        if (0 == _stricmp(path, writepath)) // not a full test, for common err
            UserMsg(hwnd,
                    "%s is the displayed image, "
                    "overwriting that is not allowed", writepath);
        else
        {
            try
            {
                WriteWindowAsBmp1(hwnd, writepath);
                TimedUserMsg(hwnd, "Wrote %s.bmp", Base(writepath));
            }
            catch(...)
            {
                UserMsg(hwnd, "Cannot write %s", writepath);
            }
        }
    }
}

} // namespace stasm
