// wasm.cpp: Windows front end to the Stasm package
//
// This code needs to be reworked.  It is a bit of a mess.

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
#include "opencv/highgui.h"
#include "../appmisc.h"
#include "findfile.h"
#include "usermsg.h"
#include "writewind.h"
#include "wasm.h"

namespace stasm
{
static const char* const APP_NAME      = "Wasm";
static const char* const WEB_ADDR      = "http://www.milbo.users.sonic.net/stasm";

// Following must match the Inno Installer script wasm.iss
// because wasm.iss is setup to remove this registry entry
// when the user uninstalls Wasm.

static const char* const REGISTRY_KEY  = "Software\\\\Wasm";
static const char* const REGISTRY_NAME = "Config";
static const char* const REGISTRY_IMG  = "ImageName";

static const int TOOLBAR_HEIGHT = 30; // in pixels

//-----------------------------------------------------------------------------
// The three windows created by Wasm are:
//
// 1. The main window (hwnd_main_g). Mostly invisible (under the toolbar and
//    child window).
//
// 2. The toolbar (toolbar_g) with its buttons,
//
// 3. The child window (hwnd_child_g).  This covers the entire main windows
//    except for the tool bar area.  We display the image in hwnd_child_g.
//
// We also create temporary windows for the help dialog box and messages.

static HWND        hwnd_main_g;        // the main window
static HWND        toolbar_g;          // toolbar

static HWND        hwnd_child_g;            // child window, displays the image
static const char* CHILD_WND = "WasmChild"; // name and class of child window

static CImage      rawimg_g;           // current image before drawing onto it
static CImage      img_g;              // current image with shape drawn
static char        img_path_g[SLEN];   // path of current image
static Shape       shape_g;            // located landmarks
static char        datadir_g[SLEN];    // directory of face detector files
static bool        searching_g;        // true while in AsmSearch

static bool        crop_g;             // crop button
static bool        gray_g;             // gray (monochrome) button

static bool        freshstart_g;       // -F flag

// cached data (makes screen update faster as you move through images in a dir)
static CImage      cache_rawimg_g;
static char        cache_path_g[SLEN];
static Shape       cache_shape_g;
static int         cache_offset_g = 1; // guess of which direction user is moving

// flags for NextFile
static bool        must_initfind_g;
static bool        done_initfind_g;

#ifdef _WIN64 // TBBUTTON bReserved has four extra bytes
  #define BUTTON_STANDARD  TBSTATE_ENABLED,TBSTYLE_BUTTON,0,0,0,0,0,0,0,0
#elif defined(_WIN32)
  #define BUTTON_STANDARD  TBSTATE_ENABLED,TBSTYLE_BUTTON,0,0,0,0
#endif

// this must be ordered as the numbers of IDM_Open etc. in wasm.h
static const TBBUTTON TOOLBAR_BUTTONS[] =
{
    3,  IDM_Open,     BUTTON_STANDARD,
    12, IDM_Blank,    BUTTON_STANDARD,
    43, IDM_PrevImg,  BUTTON_STANDARD,
    44, IDM_NextImg,  BUTTON_STANDARD,
    12, IDM_Blank,    BUTTON_STANDARD,
    12, IDM_Blank,    BUTTON_STANDARD,
    1,  IDM_Crop,     BUTTON_STANDARD,
    0,  IDM_Gray,     BUTTON_STANDARD,
    12, IDM_Blank,    BUTTON_STANDARD,
    6,  IDM_WriteImg, BUTTON_STANDARD,
    12, IDM_Blank,    BUTTON_STANDARD,
    13, IDM_Help,     BUTTON_STANDARD,
                      -1,              // -1 terminates list
};

static const char* TOOLTIPS[] =
{
    " Open an image ",                                        // IDM_Open
    " Previous image in directory \n\n Keyboard  PgUp ",      // IDM_PrevImg
    " Next image in directory \n\n Keyboard  Space or PgDn ", // IDM_NextImg
    " Crop display to face? ",                                // IDM_Crop
    " Color or gray display? ",                               // IDM_Gray
    " Write displayed image to disk ",                        // IDM_WriteImg
    " Help ",                                                 // IDM_Help
    "",                                                       // IDM_Blank
};

static void ImgPathFromUser(
    char* path,                 // out: image path got from user
    HWND  hwnd)                 // in
{
    static char path1[SLEN];    // TODO This has to be static, why?
    // On some systems, the first time it is called GetOpenFileName sometimes
    // hangs for several seconds after closing its own dialog window (possibly
    // the virus checker in action?).  So to prevent a blank window with no
    // message, we put up this "Opening file..." pacifier here.
    // TODO why is this happening?
    if (img_path_g[0] == 0)
        UserMsg(hwnd_child_g, "Opening file...");
    OPENFILENAME ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = hwnd;
    ofn.lpstrFile = path1;
    ofn.nMaxFile = sizeof(path1);
    ofn.lpstrFilter = "Images\0*.jpg;*.bmp;*.png;*.pgm;*.ppm\0All\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
    path[0] = 0;
    if (GetOpenFileName(&ofn)) // displays open file dialog box
        strcpy_s(path, SLEN, ofn.lpstrFile);
    if (img_path_g[0] == 0)
        CloseUserMsg(); // close "Opening file" pacifier message
}

HWND hCreateToolbar(
    HBITMAP&        hToolbarBmp,    // out
    const HWND      hwnd,           // in
    const TBBUTTON* toolbarButtons, // in
    int             iToolbarBmp)    // in
{
    HINSTANCE hInstance = HINSTANCE(GetWindowLongPtr(hwnd, GWLP_HINSTANCE));
    hToolbarBmp = LoadBitmap(hInstance, MAKEINTRESOURCE(iToolbarBmp));
    if (!hToolbarBmp)
        CV_Assert(!"CreateToolbar failed, no BMP");

    int button;
    for (button = 0; button < 100; button++) // calculate number of buttons
        if (toolbarButtons[button].iBitmap < 0)
            break;

    CV_Assert(button < 100);

    HWND hToolbar = CreateToolbarEx(
            hwnd,                   // parent window
            WS_VISIBLE|TBSTYLE_FLAT|TBSTYLE_TOOLTIPS,
            1,                      // child ID of this toolbar child window
            1,                      // nBitmaps
            HINSTANCE(NULL),        // hBmInst
            UINT_PTR(hToolbarBmp),  // wBMID
            toolbarButtons,         // lpButtons
            button,                // iNumButtons (including separators)
            0, 0, 0, 0,             // position etc.
            sizeof(TBBUTTON));

    if (!hToolbar)
        CV_Assert(!"CreateToolbar failed");

    return hToolbar;
}

// display a tooltip (user positioned the mouse over a toolbar button)

static void WmNotify(
    HWND,
    UINT,
    WPARAM,
    LPARAM lParam)
{
    const LPNMHDR pnmh = (LPNMHDR)lParam;
    if (pnmh->code == TTN_NEEDTEXT)
    {
        const LPTOOLTIPTEXT p = (LPTOOLTIPTEXT)lParam;
        const INT_PTR       button = p->hdr.idFrom - IDM_Open;
        if (button >= 0 && button < NELEMS(TOOLTIPS)) // paranoia
            STRCPY(p->szText, TOOLTIPS[button]);
    }
}

static void DisplayButton(HWND hToolbar, int idm, int flag)
{
    SendMessage(hToolbar, TB_CHECKBUTTON, idm,(LPARAM)MAKELONG(flag, 0));
}

static bool OpenWithShell(const char* path) // returns true if successful
{
    // TODO this is not reliable, sometimes the use has to try twice
    return INT_PTR(ShellExecute(NULL, "open", path, NULL, NULL, SW_SHOW)) >= 32;
}

// callback procedure for the help dialog window

static INT_PTR CALLBACK HelpProc(HWND hDlg, UINT msg, WPARAM wParam, LPARAM)
{
    switch (msg)
    {
        case WM_INITDIALOG:
            SetDlgItemText(hDlg, IDC_VERSION,
                ssprintf("%s %s", APP_NAME, stasm_VERSION));
            return true;
        case WM_COMMAND:
            switch (LOWORD(wParam))
            {
                case IDC_README:
                {
                    // TODO Following will only work if wasm is installed in the
                    // standard place, so won't work for non english locales with
                    // a different "Program Files".
                    const char *path1 =
                        "\"\\Program Files (x86)\\Wasm\\stasm\\Wasm-readme.html\"";
                    const char *path2 =
                        "\"\\Program Files\\Wasm\\stasm\\Wasm-readme.html\"";
                    if (!OpenWithShell(path1))
                        if (!OpenWithShell(path2))
                            UserMsg(hwnd_child_g,
                                "Cannot open readme file\n\nTried\n%s\nand\n%s",
                                path1, path2);
                    EndDialog(hDlg, 0);
                    return true;
                }
                case IDC_WEBSITE:
                    if (!OpenWithShell(WEB_ADDR))
                        UserMsg(hwnd_child_g, "Cannot open %s", WEB_ADDR);
                    EndDialog(hDlg, 0);
                    return true;
                case IDOK:
                    EndDialog(hDlg, 0);
                    return true;
            }
    }
    return false;
}

// Get the location of the data directory (which holds the face detector files).
// Do this by using the directory containing the program that is currently
// running (e.g. wasm.exe), and append ../data.
// This means that you can invoke stasm.exe from any directory and
// we still find the correct data directory.

static void DataDirFromExePath(
    char*       datadir,           // out
    const char* exepath)           // in
{
    char path[SLEN];

    if (exepath[0] == '"')
    {
        // Special handling for paths like "C:/stasm/stasm.exe"
        // Used when run as a window application from "\Program Files\stasm"

        // Copy exepath but with quotes and trailing spaces stripped
        int j = 0;
        for (int i = 1; exepath[i] && exepath[i] != '"'; i++)
            path[j++] = exepath[i];
        path[j] = 0;
    }
    else
        STRCPY(path, exepath);

    // drop base.exe

    char drive[_MAX_DRIVE], dir[_MAX_DIR];
    splitpath(path, drive, dir, NULL, NULL);
    makepath(path, drive, dir, NULL, NULL);
    char s[SLEN];
    if (path[0])
        sprintf_s(s, SLEN, "%s/../data", path);
    else
        sprintf_s(s, SLEN, "../data");

    // hack so stasm.exe can be in Release/ or Debug/ subdirectory (Visual C IDE)

    struct _stat stat;
    if (_stat(s, &stat) != 0)
    {
        char s1[SLEN];
        if (path[0])
            sprintf_s(s1, SLEN, "%s/../../data", path);
        else
            sprintf_s(s1, SLEN, "../../data");
        if (_stat(s1, &stat) != 0)
            Err("Cannot locate the data directory\n(tried %s and %s)", s, s1);
        STRCPY(s, s1);
    }
    _fullpath(datadir, s, SLEN-1);

    // force drive prefix C: to be upper case (for mingw compat with VisualC)

    if (STRNLEN(datadir, SLEN) >= 2 && datadir[1] == ':')
        datadir[0] = (char)toupper(datadir[0]);

    ConvertBackslashesToForwardAndStripFinalSlash(datadir); // for consistency
}

// Create the same window layout as last time by looking at the registry.
// Also get global toolbar button settings crop_g and gray_g.
// Init to defaults if nothing in registry or freshstart_g is set.

static void StateFromRegistry(
    int& x,                       // out
    int& y,                       // out
    int& width,                   // out
    int& height)                  // out
{
    HKEY  hkey = NULL;
    DWORD nbuf = SLEN;
    BYTE  buf[SLEN];
    int   prev_x = 0, prev_y=0, prev_width=0, prev_height=0;
    int   prev_crop = 0, prev_gray = 0;
    bool  freshstart = freshstart_g;
    char  prev_img_path[SLEN]; prev_img_path[0] = 0;

    if (!freshstart)
    {
        // get values from registry and validate them

        if (ERROR_SUCCESS !=
                RegOpenKeyEx(HKEY_CURRENT_USER, REGISTRY_KEY, 0, KEY_READ, &hkey) ||
                ERROR_SUCCESS !=
                RegQueryValueEx(hkey, REGISTRY_NAME, NULL, NULL, buf, &nbuf))
        {
            printf(
                "Cannot read registry HKEY_CURRENT_USER Key \"%s\" Name \"%s\"\n"
                "No problem, will use the default settings "
                "(probably running %s for the first time)\n",
                REGISTRY_KEY, REGISTRY_NAME, APP_NAME);
            fflush(stdout);
            freshstart = true;
        }
        else if (6 != sscanf_s((const char*)buf,
                    "%d %d %d %d %d %d",
                    &prev_x, &prev_y, &prev_width, &prev_height,
                    &prev_crop, &prev_gray) ||
                 prev_x + prev_width < 20 || prev_y + prev_height < 20 ||
                 prev_width < 50          || prev_height < 50)
        {
            printf("Cannot get values from registry HKEY_CURRENT_USER "
                   "Key \"%s\" Name \"%s\"\n"
                   "No problem, will use the default settings "
                   "(probably running %s for the first time)\n",
                   REGISTRY_KEY, REGISTRY_NAME, APP_NAME);
            fflush(stdout);
            freshstart = true;
        }
        // get image path that was used last time
        nbuf = SLEN;
        if (!freshstart &&
                ERROR_SUCCESS ==
                    RegQueryValueEx(hkey, REGISTRY_IMG, NULL, NULL, buf, &nbuf))
        {
            strncpy_s(prev_img_path,(char*)buf, SLEN);
        }
        RegCloseKey(hkey);
    }
    RECT rectWorkArea;
    SystemParametersInfo(SPI_GETWORKAREA, 0, &rectWorkArea, 0);
    if (freshstart                                               ||
            prev_width > rectWorkArea.right - rectWorkArea.left  ||
            prev_width < 20                                      ||
            prev_height > rectWorkArea.bottom - rectWorkArea.top ||
            prev_height < 20                                     ||
            prev_x + prev_width < rectWorkArea.left + 10         ||
            prev_y < rectWorkArea.top - 20)
    {
        prev_width = cvRound(.3 * (rectWorkArea.right - rectWorkArea.left));
        prev_height = cvRound(.6 * (rectWorkArea.bottom - rectWorkArea.top));
        prev_x = rectWorkArea.right - prev_width;
        prev_y = rectWorkArea.top;
        prev_crop = 0;
        prev_gray = 0;
        prev_img_path[0] = 0;
    }
    x = prev_x;
    y = prev_y;
    width = prev_width;
    height = prev_height;
    crop_g = prev_crop != 0;
    gray_g = prev_gray != 0;
    strncpy_s(img_path_g, prev_img_path, SLEN);
}

static void StateToRegistry(HWND hwndMain)
{
    RECT  rectMain;
    HKEY  hkey;
    DWORD dw; // dummy dispo argument
    char  regval[SLEN];

    GetWindowRect(hwndMain, &rectMain);

    sprintf_s(regval, SLEN, "%ld %ld %ld %ld %d %d %d",
              rectMain.left, rectMain.top,
              rectMain.right-rectMain.left,
              rectMain.bottom-rectMain.top, crop_g, gray_g);

    // no point in checking return values in func calls below
    RegCreateKeyEx(HKEY_CURRENT_USER, REGISTRY_KEY, 0, "", 0,
                   KEY_ALL_ACCESS, NULL, &hkey, &dw);
    RegSetValueEx(hkey, REGISTRY_NAME, 0, REG_SZ,
                 (CONST BYTE*)regval, STRNLEN(regval, SLEN)+1);
    RegSetValueEx(hkey, REGISTRY_IMG, 0, REG_SZ,
                 (CONST BYTE*)img_path_g, STRNLEN(img_path_g, SLEN)+1);
    RegCloseKey(hkey);
}

// tell the buttons to display themselves as depressed or not

static void DisplayButtons(void)
{
    DisplayButton(toolbar_g, IDM_Crop, crop_g);
    DisplayButton(toolbar_g, IDM_Gray, gray_g);
}

// true if red, green and blue have the same value

static bool IsGray(const CImage& img, int x, int y)
{
return img(y,x)[0] == img(y,x)[1] &&
       img(y,x)[0] == img(y,x)[2];
}

static void PossiblyIssueAlreadyGrayMsg(void)
{
    if (img_g.cols) // image loaded?
    {
        int w = img_g.cols / 2, h = img_g.rows / 2;
        CV_Assert(w > 3 && h > 3);
        if (gray_g && IsGray(rawimg_g, 0,   0) &&
                      IsGray(rawimg_g, 1,   3) &&
                      IsGray(rawimg_g, w,   h) &&
                      IsGray(rawimg_g, w+1, h) &&
                      IsGray(rawimg_g, w/2, h/2))
        {
            TimedUserMsg(hwnd_child_g, "This image is already monochrome?");
        }
    }
}

static void DisplayTitle(void)
{
    char s[SLEN];
    if (img_path_g[0] == 0)
        sprintf_s(s, SLEN, "%s version %s", APP_NAME, stasm_VERSION);
    else
    {
        char drive[_MAX_DRIVE], dir[_MAX_DIR], base[_MAX_FNAME], ext[_MAX_EXT];
        splitpath(img_path_g, drive, dir, base, ext);
        char drivedir[SLEN];
        makepath(drivedir, drive, dir, NULL, NULL);
        sprintf_s(s, SLEN, "%s%s     %d x %d     %s",
                  base, ext, img_g.rows, img_g.cols, drivedir);
    }
    SetWindowText(hwnd_main_g, s);
}

static bool IsShapeValid(void) // true if successful AsmSearch
{
    return shape_g.rows > 0;
}

// StretchDIBits fails if the image width is not divisible by 4.
// This is an (expensive) work around: pad the right side of the
// image so the width is divisible by 4.

static RGBV* Img4(
    int&         width4, // out: the new image width (often same as original)
    const CImage img)    // in
{
    RGBV* buf = (RGBV*)img.data;
    width4 = img.cols;
    if (width4 & 3) // not divisible by 4?
    {
        width4 = int((img.cols-4) / 4) * 4 + 4; // smaller int that is div by 4
        buf = (RGBV*)malloc(width4 * img.rows * sizeof(RGBV));
        if (buf == NULL)
            Err("Out of memory");
        for (int i = 0; i < img.rows; i++)
            memcpy(buf + (i * width4),
                   img.data + (i * img.cols * sizeof(RGBV)),
                   width4 * sizeof(RGBV));
    }
    return buf;
}

// display the image with the shape

static void DisplayImg(
    HDC  hdc,
    HWND hwnd)          // for hwnd_child_g
{
    RECT rect;
    GetClientRect(hwnd, &rect);

    int width4; // width of image after ensuring width is divisible by 4
    RGBV* p = Img4(width4, img_g); // ensure width is divisible by 4

    BITMAPINFO BmInfo; memset(&BmInfo.bmiHeader, 0, sizeof(BmInfo.bmiHeader));
    BmInfo.bmiHeader.biSize     = 40;
    BmInfo.bmiHeader.biWidth    = width4;
    // note the negative sign here, necessary because
    // OpenCV images are upside down wrt windows bitmaps
    BmInfo.bmiHeader.biHeight   = -img_g.rows;
    BmInfo.bmiHeader.biPlanes   = 1;
    BmInfo.bmiHeader.biBitCount = 24;

    SetStretchBltMode(hdc, COLORONCOLOR);

    StretchDIBits(hdc,
                  0, 0,                    // nxDestUpperLeft, nyDestUpperLeft
                  rect.right, rect.bottom, // nDestWidth, nDestHeight
                  0, 0,                    // xSrcLowerLeft, ySrcLowerLeft
                  img_g.cols, img_g.rows,  // nSrcWidth, nSrcHeight
                  p,                       // lpBits
                  (LPBITMAPINFO)&BmInfo,   // lpBitsInfo
                  DIB_RGB_COLORS,          // wUsage
                  SRCCOPY);                // raser operation code

    if (p != (RGBV*)img_g.data)            // Img4 allocated a new buffer?
        free(p);
}

static void DisplayNoImg(HDC hdc, HWND hwnd) // for hwnd_child_g
{
    // fill window with a solid color
    RECT rect;
    GetClientRect(hwnd, &rect);
    HBRUSH hBrush = CreateSolidBrush(RGB(64, 64, 64));
    FillRect(hdc, &rect, hBrush);
    DeleteObject(hBrush);
    DeleteObject(SelectObject(hdc, GetStockObject(SYSTEM_FONT)));
    ReleaseDC(hwnd, hdc);
}

static void AnnotateImg(void) // update img_g
{
    rawimg_g.copyTo(img_g);

    if (gray_g)
        DesaturateImg(img_g);   // convert to gray (but still RGB image)

    if (IsShapeValid())          // successful AsmSearch?
    {
        DrawShape(img_g, shape_g);
        if (crop_g)
            CropCimgToShapeWithMargin(img_g, shape_g);
    }
    else
        UserMsg(hwnd_child_g, "No face found");
}

// TODO: If you pass an Image (not a CImage) to this function you will get:
// OpenCV Error: The function/feature is not implemented in matrix.cpp.
// So in that case we never even get to the CV_Assert below.

static Image AsImage(const CImage& cimg) // convert a CImage to an Image
{
    CV_Assert(cimg.depth() == CV_8U && cimg.channels() == 3);
    Image img; cvtColor(cimg, img, CV_BGR2GRAY);
    return img;
}

static const Shape AsmSearch(
    const CImage& cimg,                  // in
    const char*   path)                  // in
{
    Image img(AsImage(cimg));

    if (!stasm_open_image((const char*)img.data, img.cols, img.rows, path,
                          0 /*multi*/, 25 /*minwidth*/))
        Err("stasm_open_image failed:  %s", stasm_lasterr());

    int foundface;
    float landmarks[2 * stasm_NLANDMARKS]; // x,y coords
    if (!stasm_search_auto(&foundface, landmarks))
        Err("stasm_search_auto failed: %s", stasm_lasterr());

    return foundface? LandmarksAsShape(landmarks): MAT(0,0);
}

bool IsFileReadable(const char* path)
{
    FILE* file;
    if (fopen_s(&file, path, "r") == 0)
    {
        fclose(file);
        return true;
    }
    return false;
}

static void OpenImg(void)
{
    if (searching_g) // prevent recursion e.g. user clicks NextImg while in here
        return;
    searching_g = true;
    if (strcmp(cache_path_g, img_path_g) == 0 && cache_rawimg_g.data)
    {
        // cached img is available
        STRCPY(img_path_g, cache_path_g);
        cache_rawimg_g.copyTo(rawimg_g);
        cache_shape_g.copyTo(shape_g);
        AnnotateImg(); // update img_g
    }
    else
    {
        // cached img is not available
        // readable check needed because imread crashes if file not readable TODO why?
        bool readable = IsFileReadable(img_path_g);
        if (readable)
            rawimg_g = cv::imread(img_path_g, CV_LOAD_IMAGE_COLOR);
        if (!readable || !rawimg_g.data)
        {
            char s[SLEN]; sprintf_s(s, SLEN, "Cannot read %s", img_path_g);
            UserMsg(hwnd_child_g, s);
            img_path_g[0] = 0;
        }
        else
        {
            shape_g = AsmSearch(rawimg_g, img_path_g);
            AnnotateImg(); // update img_g
        }
    }
    searching_g = false;
}

static void PossiblyInitCacheImg(void)
{
    if (done_initfind_g)
    {
        char path[SLEN];
        NextFilePeek(path, cache_offset_g);        // get path of next image
        if (path[0] && strcmp(path, cache_path_g)) // changed?
        {
            STRCPY(cache_path_g, path);
            // readable check needed because imread crashes if file not readable TODO why?
            bool readable = IsFileReadable(cache_path_g);
            if (readable)
                cache_rawimg_g = cv::imread(cache_path_g, CV_LOAD_IMAGE_COLOR);
            if (!readable || !cache_rawimg_g.data)
                STRCPY(cache_path_g, "nonesuch")
            else
                cache_shape_g = AsmSearch(cache_rawimg_g, cache_path_g);
        }
    }
}

static void IdmOpen(void)
{
    char path[SLEN];
    ImgPathFromUser(path, hwnd_main_g);
    if (path[0])                     // valid path?
    {
        STRCPY(img_path_g, path);
        OpenImg();
        if (img_path_g[0]) // OpenImg succesful?
        {
            must_initfind_g = true;
            done_initfind_g = false;
            cache_offset_g = 1;
        }
    }
}

static void PossiblyInitFindFile(void)
{
    if (must_initfind_g)
    {
        InitFindFile(img_path_g, ".jpg;.bmp;.png;.pgm;.ppm");
        must_initfind_g = false;
        done_initfind_g = true;
    }
}

static void IdmNextImg(void)
{
    PossiblyInitFindFile();
    bool many = NextFile(img_path_g, 1); // init img_path_g
    cache_offset_g = 1;
    if (!many)
        TimedUserMsg(hwnd_child_g, "Only one image in the current directory");
    else
        OpenImg();
}

static void IdmPrevImg(void)
{
    PossiblyInitFindFile();
    bool many = NextFile(img_path_g, -1); // init img_path_g
    cache_offset_g = -1;
    if (!many)
        TimedUserMsg(hwnd_child_g, "Only one image in the current directory");
    else
        OpenImg();
}

static void IdmFirstImg(void)
{
    PossiblyInitFindFile();
    bool many = NextFile(img_path_g, FINDFILE_FIRST);
    cache_offset_g = 1;
    if (!many)
        TimedUserMsg(hwnd_child_g, "Only one image in the current directory");
    else
        OpenImg();
}

static void IdmLastImg(void)
{
    PossiblyInitFindFile();
    bool many = NextFile(img_path_g, FINDFILE_LAST);
    cache_offset_g = 1;
    if (!many)
        TimedUserMsg(hwnd_child_g, "Only one image in the current directory");
    else
        OpenImg();
}

static void Wm_Create(HWND hwnd)
{
    // the double typecast is necessary to not get a warning in 64 bit builds
    HINSTANCE hInstance =
        HINSTANCE(LONG_PTR(GetWindowLongPtr(hwnd, GWLP_HINSTANCE)));

    hwnd_child_g = CreateWindow(CHILD_WND,
                                NULL,                        // window caption
                                WS_CHILDWINDOW | WS_VISIBLE, // window style
                                0,                           // x position
                                0,                           // y position
                                1,                           // x size
                                1,                           // y size
                                hwnd,                        // parent wind handle
                                NULL,                        // window menu handle
                                hInstance,                   // program inst handle
                                NULL);                       // creation paramss

    if (NULL == hwnd_child_g)
        CV_Assert(!"CreateWindow failed for child window");
}

static void Wm_Size(HWND, LPARAM lParam)
{
    SendMessage(toolbar_g, TB_AUTOSIZE, 0, 0L);

    int child_width = LOWORD(lParam); // width of parent

    // height of parent minus toolbar
    int child_height = MAX(1, HIWORD(lParam)- TOOLBAR_HEIGHT);

    MoveWindow(hwnd_child_g, 0, TOOLBAR_HEIGHT,
               child_width, child_height, true);
}

static void ChildWm_Paint(HWND hwnd)
{
    DisplayTitle();
    DisplayButtons();

    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hwnd, &ps);
    if (img_path_g[0])
        DisplayImg(hdc, hwnd);
    else
        DisplayNoImg(hdc, hwnd);  // no image to display, so display gray
    EndPaint(hwnd, &ps);

    // while user is looking at the image, get next image to save time later
    PossiblyInitFindFile();
    PossiblyInitCacheImg();
}

static bool IsImgLoaded(void)
{
    const bool loaded = img_path_g[0] != 0;
    if (!loaded)
        UserMsg(hwnd_child_g, "No image.  First open an image.");
    return loaded;
}

void Wm_Command(HWND hwnd, UINT, WPARAM wParam, LPARAM)
{
    CloseUserMsg();
    switch (wParam)
    {
        case IDM_Open:
            IdmOpen();
            break;

        case IDM_PrevImg:
            if (IsImgLoaded())
                IdmPrevImg();
            break;

        case IDM_NextImg:
            if (IsImgLoaded())
                IdmNextImg();
            break;

        case IDM_Crop:
            crop_g ^= 1;
            if (img_path_g[0])
                AnnotateImg();
            break;

        case IDM_Gray:
            gray_g ^= 1;
            if (img_path_g[0])
                AnnotateImg();
            PossiblyIssueAlreadyGrayMsg();
            break;

        case IDM_WriteImg:
            if (IsImgLoaded())
                WriteWindowAsBmp(hwnd_child_g, img_path_g, APP_NAME);
            break;

        case IDM_Help:
        {
            HINSTANCE hInstance = HINSTANCE(GetWindowLongPtr(hwnd, GWLP_HINSTANCE));
            if (DialogBox(hInstance, "HelpDlg", hwnd, HelpProc) < 0)
                UserMsg(hwnd_child_g, "DialogBox failed");
            break;
        }
        case IDM_Blank:
            break;

        default:
            Err("Wm_Command bad param %u", wParam);
            break;
    }
    // InvalidateRect triggers a repaint i.e. it causes ChildWndProc WM_PAINT
    InvalidateRect(hwnd_child_g, NULL, false);
}

static void Wm_Keydown(HWND, WPARAM wParam)
{
    switch (wParam)
    {
        case ' ':                   // space
        case VK_NEXT:               // page down
            CloseUserMsg();
            IdmNextImg();
            break;
        case VK_PRIOR:              // page up
            CloseUserMsg();
            IdmPrevImg();
            break;
        case VK_HOME:               // first image
            CloseUserMsg();
            IdmFirstImg();
            break;
        case VK_END:                // last image
            CloseUserMsg();
            IdmLastImg();
            break;
    }
    InvalidateRect(hwnd_child_g, NULL, false); // trigger ChildWndProc WM_PAINT
}

static LRESULT CALLBACK ChildWndProc( // for hwnd_child_g
    HWND   hwnd,
    UINT   nMsg,
    WPARAM wParam,
    LPARAM lParam)
{
    switch (nMsg)
    {
        case WM_PAINT:
            ChildWm_Paint(hwnd);    // this updates the picture the user sees
            return 0;
        case WM_KEYDOWN:            // user hit a key
            Wm_Keydown(hwnd, wParam);
            return 0;
    }
    return DefWindowProc(hwnd, nMsg, wParam, lParam);
}

static LRESULT CALLBACK WndProc( // for hwnd_main_g
    HWND   hwnd,
    UINT   nMsg,
    WPARAM wParam,
    LPARAM lParam)
{
    switch (nMsg)
    {
        case WM_CREATE:
            Wm_Create(hwnd);
            return 0;
        case WM_SIZE:
            Wm_Size(hwnd, lParam);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_CLOSE:
            StateToRegistry(hwnd_main_g);
            break;          // call default window close handler
        case WM_SETFOCUS:   // set focus to child window
            SetFocus(hwnd_child_g);
            return 0;
        case WM_COMMAND:    // user clicked a toolbar button
            Wm_Command(hwnd, nMsg, wParam, lParam);
            return 0;
        case WM_NOTIFY:     // possibly display a tooltip
            WmNotify(hwnd, nMsg, wParam, lParam);
            return 0;
        case WM_KEYDOWN:    // user hit a key
            Wm_Keydown(hwnd, wParam);
            return 0;
    }
    return DefWindowProc(hwnd, nMsg, wParam, lParam);
}

static void ProcessCmdLine(LPSTR lpCmdLine)
{
    static const char* const usage =
    "Usage:  wasm  [-F]\n"
    "\n"
    "-F\tFresh start (ignore saved window positions etc.)\n"
    "\n"
    "Wasm version %s\n";  // %s is stasm_VERSION

    static const char* whitespace = " \t";
    char* next_token;
    char* token = strtok_s(lpCmdLine, whitespace, &next_token);
    while (token != NULL)
    {
        if (token[0] == '-')
        {
            if (token[1] == 0)
                Err(usage, stasm_VERSION);
            switch (token[1])
            {
                case 'F':
                    freshstart_g = true;
                    break;
                default:    // bad flag
                    Err(usage, stasm_VERSION);
                    break;
            }
        }
        else
            Err(usage, stasm_VERSION);
        token = strtok_s(NULL, whitespace, &next_token);
    }
}

static void Init(
    HINSTANCE hInstance,
    LPSTR     lpCmdLine)
{
    // shutdown if this program is running already
    CreateMutex(NULL, true, APP_NAME);
    if (GetLastError() == ERROR_ALREADY_EXISTS)
        Err("%s is running already", APP_NAME);

    ProcessCmdLine(lpCmdLine);
    DataDirFromExePath(datadir_g, GetCommandLine());

    if (!stasm_init(datadir_g, 0 /*trace*/))
        Err("stasm_init failed %s", stasm_lasterr());

    WNDCLASSEX wndclass;
    wndclass.lpszClassName = APP_NAME;
    wndclass.cbSize        = sizeof(wndclass);
    wndclass.style         = CS_HREDRAW | CS_VREDRAW;
    wndclass.lpfnWndProc   = WndProc;
    wndclass.cbClsExtra    = 0;
    wndclass.cbWndExtra    = 0;
    wndclass.hInstance     = hInstance;
    wndclass.hbrBackground = (HBRUSH)GetStockObject(LTGRAY_BRUSH);
    wndclass.lpszMenuName  = NULL;
    wndclass.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wndclass.hIcon         = LoadIcon(hInstance, APP_NAME);
    wndclass.hIconSm       = LoadIcon(hInstance, APP_NAME);

    if (!RegisterClassEx(&wndclass))
        CV_Assert(!"RegisterClass failed");

    // Create the class for the child (image) window, we will create
    // the actual window later in when the main window gets a WM_CREATE

    wndclass.lpfnWndProc   = ChildWndProc;
    wndclass.hIcon         = NULL;
    wndclass.hIconSm       = NULL;
    wndclass.hbrBackground = NULL; // we do all our own painting
    wndclass.lpszClassName = CHILD_WND;

    if (!RegisterClassEx(&wndclass))
        CV_Assert(!"RegisterClass failed");

    // use the same window layout as last time by looking at registry
    int x, y, width, height; StateFromRegistry(x, y, width, height);

    hwnd_main_g = CreateWindow(APP_NAME,
                              APP_NAME,            // window caption
                              WS_OVERLAPPEDWINDOW, // window style
                              x,                   // x position
                              y,                   // y position
                              width,               // x size
                              height,              // y size
                              NULL,                // parent window handle
                              NULL,                // window menu handle
                              hInstance,          // program instance handle
                              NULL);               // creation parameters

    if (!hwnd_main_g)
        CV_Assert(!"CreateWindow failed");

    HBITMAP hToolbarBmp;
    toolbar_g = hCreateToolbar(hToolbarBmp,
                               hwnd_main_g, TOOLBAR_BUTTONS, IDR_TOOLBAR_BITMAP);

    ShowWindow(hwnd_main_g, SW_SHOW);
    UpdateWindow(hwnd_main_g);
    DisplayButtons();

    if (img_path_g[0])    // got previous image name from the registry?
    {
        OpenImg();
        if (img_path_g[0]) // OpenImg succesful?
        {
            must_initfind_g = true;
            done_initfind_g = false;
        }
        InvalidateRect(hwnd_child_g, NULL, false); // trigger ChildWndProc WM_PAINT
    }
    else
        UserMsg(hwnd_child_g, "Click the open button above");
}

static void WinMain1(
    HINSTANCE   hInstance,
    const LPSTR lpCmdLine)
{
    Init(hInstance, lpCmdLine);
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

} // namespace stasm

// This application calls Stasm's internal routines.  Thus we need to catch a
// potential throw from Stasm's error handlers.  Hence the try/catch code below.

int WINAPI WinMain(
    HINSTANCE hInstance,
    HINSTANCE,
    LPSTR     lpCmdLine,
    int)
{
    stasm::CatchOpenCvErrs();
    try
    {
        stasm::WinMain1(hInstance, lpCmdLine);
    }
    catch(...)
    {
        // a call was made to Err or a CV_Assert failed
        MessageBox(NULL, stasm_lasterr(), "Wasm Error", MB_OK);
        return 1;   // failure
    }
    return 0;       // success
}
