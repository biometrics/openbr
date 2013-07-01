// stasm_regex.h:
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_REGEX_H
#define STASM_REGEX_H

// define HAVE_REGEX
#if _MSC_VER && _MSC_VER >= 1500 // vc9 or greater
  #define HAVE_REGEX 1
  #include <regex>
  #if _MSC_VER >= 1600           // vc10 or greater
    using std::regex;
    using std::regex_constants::icase;
    using std::regex_search;
  #else
    using std::tr1::regex;
    using std::tr1::regex_constants::icase;
    using std::tr1::regex_search;
  #endif
#else
  #define HAVE_REGEX 0           // gcc doesn't yet support regexs
  typedef std::string regex;     // fakery
#endif

namespace stasm
{
const regex CompileRegex(        // convert string sregex to case indepdent regex
    const char* sregex);         // in

bool MatchRegex(                 // true if regex is in the given string
    const string& s,             // in: the string
    const regex&  re);           // in: a compiled regex

void RegexToLegalFilename(       // Example:   [^ab]|c   becomes   ~ab_c
    char* s);                    // io: the regex

} // namespace stasm
#endif // STASM_REGEX_H
