// usermsg.h: Routines for displaying messages in a Windows environment
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_USERMSG_H
#define STASM_USERMSG_H

namespace stasm
{
void CloseUserMsg(void);

void TimedUserMsg(HWND hwnd, const char* format, ...);

void UserMsg(HWND hwnd, const char* format, ...);

} // namespace stasm
#endif // STASM_USERMSG_H
