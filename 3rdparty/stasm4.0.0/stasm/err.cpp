// err.cpp: error handling for Stasm
//
// Don't call internal Stasm functions (i.e. functions in the Stasm namespace
// when not in a try block, because Stasm's error handler (Err) raises an
// exception. You need to catch that exception to avoid a messageless crash.
//
// Your code should look like this (see e.g. stasm_lib.cpp and stasm_main.cpp):
//
//         CatchOpenCvErrs(); // tell Stasm to handle CV_Assert as Stasm errors
//         try
//         {
//             ... your code which calls Stasm's internal functions ...
//         }
//         catch(...)
//         {
//             // a call was made to Err or a CV_Assert failed
//             printf("\n%s\n", stasm_lasterr());
//             exit(1);
//         }
//         UncatchOpenCvErrs(); // restore OpenCV handler to its previous state
//
// Note that the stasm library function (i.e. the functions prefixed
// by "stasm_") use try blocks internally, and code that calls
// them doesn't have to worry about the above exception.
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"

namespace stasm
{
static char err_g[SBIG]; // err msg saved for retrieval by LastErr and stasm_lasterr

static vector<cv::ErrorCallback> stack_g(10); // stack of err handlers, 10 is generous

static int istack_g;                          // index into stack_g

//-----------------------------------------------------------------------------

static void PossiblyEnterDebugger(void)
{
#if _DEBUG
    // requires you to be in a debugger or have set up a just-in-time debugger
    printf("\n%s\nDEBUG is true so forcing entry to the debugger\n", err_g);
    fflush(stdout);
    static volatile int* p = 0;
    *p = 99;
#endif // _DEBUG
}

// This gets called during OpenCV error handling e.g. if a CV_Assert fails.
// Save the error info in our global string err_g.

static int CV_CDECL CvErrorCallbackForStasm(
    int         code,      // translated to a string e.g. "Assertion failed"
    const char* ,          // unused here
    const char* err_msg,   // e.g. the contents of the line where assert failed
    const char* file_name, // filename where error occurred (if available)
    int         line,      // line number where error occurred
    void*       )          // unused here
{
    if (err_g[0])
    {
        // Recursive, we are already processing an error.
        // Not really an issue, only first error will be reported via LastErr.
        printf("\nNested error in CvErrorCallbackForStasm\n"
               "  Current error: %.80s\n  New error:     %.80s\n",
               err_g, err_msg);
    }
    else
    {
        char temp[SBIG]; // temporary string needed because err_msg may be err_g
        const char* errmsg = cvErrorStr(code);
        if (file_name && file_name[0])
            sprintf(temp, "%s(%d) : %s : %s",
                    BaseExt(file_name), line, errmsg, err_msg);
        else
            sprintf(temp, "OpenCV %s : %s", errmsg, err_msg);

        STRCPY(err_g, temp);
    }
    PossiblyEnterDebugger();
    return 0;
}

void CatchOpenCvErrs(void) // makes CV_Assert work with LastErr and stasm_lasterr
{
    err_g[0] = 0;
    cv::ErrorCallback prev = cv::redirectError(CvErrorCallbackForStasm);
    if (istack_g < NSIZE(stack_g))
        stack_g[istack_g++] = prev;
    else // should never get here (CatchErr regions nested too deeply)
        printf("\nCallback stack overpush\n");
}

void UncatchOpenCvErrs(void) // restore handler that was active before CatchOpenCvErrs
{
    if (istack_g > 0)
        cv::redirectError(stack_g[--istack_g]);
    else // should never get here (call to UncatchErr without matching CatchErr)
        printf("\nCallback stack overpop\n");
}

void Err(const char* format, ...) // args like printf, throws an exception
{
    if (err_g[0])
    {
        // Recursive, we are already processing an error.
        // Ok, only first error will be reported via LastErr.
        // This happens if Err is called to report a stasm_search_auto fail.
    }
    else
    {
        char s[SBIG]; // temporary needed because format or ... may be err_g
        va_list args;
        va_start(args, format);
        VSPRINTF(s, format, args);
        va_end(args);
        STRCPY(err_g, s);
    }
    PossiblyEnterDebugger();
    throw "Err"; // does not matter what we throw, will be caught by global catch
}

const char* LastErr(void) // return the last error message, called by stasm_lasterr
{
    if (!err_g[0]) // no error message?
    {
        // Should never get here unless someone calls LastErr or
        // stasm_lasterr incorrectly (i.e. when there has been no error).
        //
        // TODO But in fact we do actually get here if cv::fastMalloc fails
        // (within OpenCV) when allocating a small amount of memory (say 10 bytes
        // large amounts are ok).  It seems that when there is very little memory
        // remaining, OpenCV does not handle exceptions properly (an exception
        // is raised but the OpenCV error callback function is not called).
        // To reproduce, put the following in your code:
        // volatile void *p; while (1) p = cv::fastMalloc(10);

        STRCPY(err_g, "Illegal call to LastErr");
    }
    return err_g;
}

} // namespace stasm
