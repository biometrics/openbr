// print.cpp: printing and logging utilities for the Stasm library
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"
#include <sys/stat.h>

namespace stasm
{
bool print_g;   // true to allow output to stdout (but error msgs always printed)

bool trace_g;   // true to trace Stasm internal operation

static FILE* logfile_g;  // lprintfs go to this log file as well as stdout

//-----------------------------------------------------------------------------

// Open the log file. After this, when you call lprintf, you print to the log
// file (as well as to stdout).  This inits the global variable logfile_g.

void OpenLogFile(void)  // also inits the global variable logfile_g
{
    if (!logfile_g)
    {
        static const char* const path = "stasm.log";
        if (print_g)
            printf("Opening %s\n", path);
        logfile_g = fopen(path, "wt");
        if (!logfile_g)
            Err("Cannot open \"%s\"", path);
        // check that we can write to the log file
        if (fputs("log file\n", logfile_g) < 0)
            Err("Cannot write to \"%s\"", path);
        rewind(logfile_g); // rewind so above test msg is not in the log file
    }
}

// Like printf but only prints if print_g flag is set,
// and also prints to the log file if it is open.

void lprintf(const char* format, ...)   // args like printf
{
    (void) format;
    /*
    if (print_g)
    {
        char s[SBIG];
        va_list args;
        va_start(args, format);
        VSPRINTF(s, format, args);
        va_end(args);
        lputs(s);
    }*/
}

// Like printf but prints to the log file only (and not to stdout).
// Used for detailed stuff that we don't usually want to see.

void logprintf(const char* format, ...) // args like printf
{
    (void) format;
    /*
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
    }*/
}

// Like puts but prints to the log file as well if it is open,
// and does not append a newline.

void lputs(const char* s)
{
    (void) s;
    /*
    printf("%s", s);
    fflush(stdout);     // flush so if there is a crash we can see what happened
    logprintf("%s", s);*/
}


} // namespace stasm
