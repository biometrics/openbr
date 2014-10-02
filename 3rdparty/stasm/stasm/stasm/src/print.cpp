// print.cpp: printing and logging utilities for the Stasm library
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "print.h"
#include "err.h"
#include "misc.h"

#include <stdio.h>
#include <sys/stat.h>
#include <stdarg.h>

namespace stasm
{
bool print_g;   // true to allow output to stdout (but error msgs always printed)

bool trace_g;   // true to trace Stasm internal operation

static FILE* logfile_g;  // lprintfs go to this log file as well as stdout

//-----------------------------------------------------------------------------

// Open the log file. After this, when you call lprintf, you print to the log
// file (as well as to stdout).  This inits the global variable logfile_g.

void OpenLogFile(     // also inits the global variable logfile_g
    const char* path) // in: log file path, default is "stasm.log"
{
    if (!logfile_g)
    {
        if (print_g)
            printf("Generating %s\n", path);
        logfile_g = fopen(path, "wb");
        if (!logfile_g)
            Err("Cannot open \"%s\"", path);
        // check that we can write to the log file
        if (fputs("log file\n", logfile_g) < 0)
            Err("Cannot write to \"%s\"", path);
        rewind(logfile_g); // rewind so above test msg is not in the log file
    }
}

// Like printf but only prints if print_g flag is set.
// Also prints to the log file if it is open (regardless of print_g).

void lprintf(const char* format, ...)   // args like printf
{
    char s[SBIG];
    va_list args;
    va_start(args, format);
    VSPRINTF(s, format, args);
    va_end(args);
    if (print_g)
    {
        printf("%s", s);
        fflush(stdout); // flush so if there is a crash we can see what happened
    }
    if (logfile_g)
    {
        // we don't check fputs here, to prevent recursive calls and msgs
        fputs(s, logfile_g);
        fflush(logfile_g);
    }
}

// Like printf but prints to the log file only (and not to stdout).
// Used for detailed stuff that we don't usually want to see.

void logprintf(const char* format, ...) // args like printf
{
    if (logfile_g)
    {
        char s[SBIG];
        va_list args;
        va_start(args, format);
        VSPRINTF(s, format, args);
        va_end(args);
        // we don't check fputs here, to prevent recursive calls and msgs
        fputs(s, logfile_g);
        fflush(logfile_g);
    }
}

// Like lprintf but always prints even if print_g is false.

void lprintf_always(const char* format, ...)
{
    char s[SBIG];
    va_list args;
    va_start(args, format);
    VSPRINTF(s, format, args);
    va_end(args);
    printf("%s", s);
    fflush(stdout); // flush so if there is a crash we can see what happened
    if (logfile_g)
    {
        // we don't check fputs here, to prevent recursive calls and msgs
        fputs(s, logfile_g);
        fflush(logfile_g);
    }
}

// Like puts but prints to the log file as well if it is open,
// and does not append a newline.

void lputs(const char* s)
{
    printf("%s", s);
    fflush(stdout);     // flush so if there is a crash we can see what happened
    logprintf("%s", s);
}

// Print message only once on the screen, and only 100 times to the log file.
// This is used when similar messages could be printed many times and it
// suffices to let the user know just once.  By convention the message is
// printed followed by "..." so the user knows that just the first message
// was printed.  The user can look in the log file for further messages if
// necessary (but we print only 100 times to the log file --- else all the
// log prints make tasm slow).

void PrintOnce(
    int&        printed,     // io: zero=print, nonzero=no print
    const char* format, ...) // in: args like printf
{
    char s[SBIG];
    va_list args;
    va_start(args, format);
    VSPRINTF(s, format, args);
    va_end(args);
    if (printed == 0 && print_g)
    {
        printed = 1;
        printf("%s", s);
        fflush(stdout); // flush so if there is a crash we can see what happened
    }
    if (printed < 100 && logfile_g)
    {
        fputs(s, logfile_g);
        fflush(logfile_g);
        printed++;
        if (printed == 100)
            logprintf("no more prints of the above message (printed == 100)\n");
    }
}

} // namespace stasm
