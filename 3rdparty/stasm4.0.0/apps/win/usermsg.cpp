// usermsg.cpp: Routines for displaying messages in a Windows environment

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

namespace stasm
{
static const char* const CLASS_NAME = "UserMsg";

static const int USER_MSG_TIMER = 10;

static HWND hwnd_g;      // the window used to display the user mesage

static char msg_g[SBIG]; // the displayed user message

//-----------------------------------------------------------------------------

static LRESULT CALLBACK TimedUserMsg_WndProc(
    HWND   hwnd,   // in
    UINT   nMsg,   // in
    WPARAM wParam, // in
    LPARAM lParam) // in
{
    switch (nMsg)
    {
        case WM_PAINT:
        {
            // Figure out the size of the window necessary for the string.
            // We get text size, and size the window accordingly.
            PAINTSTRUCT ps; HDC hdc = BeginPaint(hwnd, &ps);
            SIZE size;
            // TODO GetTextExtentPoint32 returns too big cx value.  This is
            // a work around.  Figure out length of longest line, use that.
            int i, j = 0, len = 0, nlines = 0;
            for (i = 0; msg_g[i]; i++)
                if (msg_g[i] == '\n')
                {
                    j = 0;
                    nlines++;
                }
                else if (++j > len)
                    len = j;
            char s[SLEN];
            // fill s with len Xs
            for (i = 0; i < len; i++)
                s[i] = 'X';
            s[i] = 0;
            GetTextExtentPoint32(hdc, s, len, &size);
            RECT rectMain; GetClientRect(hwnd, &rectMain);
            MoveWindow(hwnd, 30, 30, // x,y pos
                       size.cx + 20, nlines * size.cy + 40, true);
            // must match hbrBackground in RegisterClassEx
            SetBkColor(hdc, RGB(255, 255, 255));
            RECT rect; GetClientRect(hwnd, &rect); rect.top += 10;
            DrawText(hdc, msg_g, -1, &rect,
                     DT_CENTER | DT_VCENTER | DT_EXPANDTABS);
            DeleteObject(SelectObject(hdc, GetStockObject(SYSTEM_FONT)));
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_TIMER:
            KillTimer(hwnd, USER_MSG_TIMER);
            PostMessage(hwnd, WM_CLOSE, 0, 0); // self destroy
            return 0;

        case WM_LBUTTONDOWN:    // mouse click in timed msg window makes it go away
        case WM_MBUTTONDOWN:    // mouse middle click
        case WM_RBUTTONDOWN:    // mouse right click
            // TODO We never get here.
            PostMessage(hwnd, WM_CLOSE, 0, 0); // self destroy
            return 0;

        case WM_DESTROY:
            KillTimer(hwnd, USER_MSG_TIMER); // release resource
            hwnd_g = NULL;
            break;
    }
    return DefWindowProc(hwnd, nMsg, wParam, lParam);
}

static void InitTimedUserMsgWindow(
    HWND hwnd, // in
    int  ms)   // in
{
    HINSTANCE hInstance = HINSTANCE(GetWindowLongPtr(hwnd, GWLP_HINSTANCE));
    CV_Assert(hInstance); // fails if you call UserMsg before hwnd is initialized
    static bool firsttime = true;
    if (firsttime)
    {
        firsttime = false;

        WNDCLASSEX wndclass;
        wndclass.cbSize        = sizeof(wndclass);
        wndclass.style         = CS_HREDRAW | CS_VREDRAW;
        wndclass.lpfnWndProc   = TimedUserMsg_WndProc;
        wndclass.cbClsExtra    = 0;
        wndclass.cbWndExtra    = 0;
        wndclass.hInstance     = hInstance;
        wndclass.hCursor       = LoadCursor(NULL, IDC_CROSS);
        wndclass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
        wndclass.lpszMenuName  = NULL;
        wndclass.lpszClassName = CLASS_NAME;
        wndclass.hIcon         = NULL;
        wndclass.hIconSm       = NULL;

        if (!RegisterClassEx(&wndclass))
            CV_Assert(0 && "RegisterClass failed");
    }
    // create a temporary child window
    hwnd_g = CreateWindow(CLASS_NAME,             // class name
                          "",                     // window caption
                          WS_CHILD | WS_BORDER,   // window style
                          0,                      // x position
                          0,                      // y position
                          1,                      // x size
                          1,                      // y size
                          hwnd,                   // parent window handle
                          NULL,                   // window menu handle
                          hInstance,
                          NULL);                  // creation parameters
    CV_Assert(hwnd_g);
    SetTimer(hwnd_g, USER_MSG_TIMER, ms, NULL);
    ShowWindow(hwnd_g, SW_SHOW);
    UpdateWindow(hwnd_g);
}

static void TimedUserMsg1(
    HWND hwnd,             // in
    int  ms)               // in: time in milliseconds
{
    if (hwnd_g) // already displaying a message?
    {
        // overwrite existing msg with a new one but use the existing window
        SetTimer(hwnd_g, USER_MSG_TIMER, ms, NULL);
        InvalidateRect(hwnd, NULL, false); // trigger repaint of main window
        return;
    }
    else
        InitTimedUserMsgWindow(hwnd, ms);
}

// This allows user action like mouse clicks to immediately
// close the timed msg window.
//
// TODO This is not quite right and causes a flicker when you click a
// toolbar button which causes a TimedUserMsg (because we invoke
// TimedUserMsg and then immediately invoke this?).

void CloseUserMsg(void)
{
    if (hwnd_g) // displaying a message?
        SetTimer(hwnd_g, USER_MSG_TIMER, 0, NULL);
}

// Put up a message for three seconds.
// Any key stroke or mouse click will make the message go away.
// Typically used for informative messages e.g. "Saved face".

void TimedUserMsg(
    HWND        hwnd,   // in
    const char* format, // in: args like printf
                ...)    // in
{
    va_list args;
    va_start(args, format);
    VSPRINTF(msg_g, format, args);
    va_end(args);
    msg_g[0] = char(toupper(msg_g[0]));
    TimedUserMsg1(hwnd, 3000); // 3000 ms
}

// Put up a message "permanently" (actually for a ten thousand seconds).
// Any key stroke or mouse click will make the message go away.
// Typically used for minor error messages e.g. "No face found"
// (serious errors should use Err to put up a message box).

void UserMsg(
    HWND        hwnd,   // in
    const char* format, // in: args like printf
                ...)    // in
{
    va_list args;
    va_start(args, format);
    VSPRINTF(msg_g, format, args);
    va_end(args);
    msg_g[0] = char(toupper(msg_g[0]));
    TimedUserMsg1(hwnd, int(1e7)); // ten thousand seconds
}

} // namespace stasm
