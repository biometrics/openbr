// stasm_regex.cpp:
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"
#include "stasm_regex.h"

namespace stasm
{
static bool AlphaNumeric(char c)
{
    return (c >= '0' && c <= '9') ||
           (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z');
}

void RegexToLegalFilename( // Example:  [^ab]|c   becomes   ~ab_c
    char* s)               // io
{
    if (s[0])
    {
        int j = 0;
        for (int i = 0; s[i]; i++)
        {
            const char c = s[i];
            if (AlphaNumeric(c) || c == '.' || c == '-' || c == '_')
                s[j++] = c;
            else if (c == '*')
                s[j++] = 'S';
            else if (c == '|')
                s[j++] = '_';
            else if (c == '^')
                s[j++] = '~';
        }
        s[j] = 0;
    }
}

#if HAVE_REGEX

const regex CompileRegex(     // convert string sregex to case independent regex
    const char* sregex)       // in
{
    try // do a dummy regex compile to check the regex syntax
    {
        regex temp(sregex, icase);
    }
    catch(...)
    {
        Err("Bad regular expression \"%s\"", sregex);
    }
    return regex(sregex, icase);
}

bool MatchRegex(              // true if regex is in the given string
    const string& s,          // in: the string
    const regex&  re)         // in: a compiled regex
{
    return regex_search(s.begin(), s.end(), re);
}

#else  // not HAVE_REGEX (so use case independent string comparisons)

const regex CompileRegex(     // convert string sregex to case independent "regex"
    const char* sregex)       // in
{
    char s1[SBIG];
    strcpy(s1, sregex);

    // does sregex include any regex special characters?
    for (int i = 0; s1[i]; i++)
    {
        if (!AlphaNumeric(s1[i]) && s1[i] != '_')
        {
            printf("\nWarning: Only plain strings (not regexs) are supported in %s\n", sregex);
            break;
        }
    }
    ToLowerCase(s1);
    return string(s1);
}

bool MatchRegex(              // true if regex is in the given string
    const string& s,          // in: the string
    const regex& re)          // in: the compiled regex (actually just a string)
{
    char s1[SBIG];
    strcpy(s1, s.c_str());
    ToLowerCase(s1);
    return strstr(s1, re.c_str()) != NULL;
}

#endif // not HAVE_REGEX

} // namespace stasm
