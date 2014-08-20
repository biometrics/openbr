// print.h: printing and logging utilities for the Stasm library
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_PRINT_H
#define STASM_PRINT_H

namespace stasm
{
extern bool print_g;   // true to allow output to stdout

extern bool trace_g;   // true to trace Stasm internal operation

void OpenLogFile(void);
void lputs(const char* s);
void lprintf(const char* format, ...); // args like printf
void logprintf(const char* format, ...);

} // namespace stasm
#endif // STASM_PRINT_H
