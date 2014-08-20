// err.h: error handling for Stasm
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_ERR_H
#define STASM_ERR_H

namespace stasm
{
void CatchOpenCvErrs(void);
void UncatchOpenCvErrs(void);
void Err(const char* format, ...);
const char* LastErr(void); // return the last error message

} // namespace stasm
#endif // STASM_ERR_H
