// mdiff.cpp: print difference between two files,
//            ignoring text in either file between []
//
// This prints the FIRST difference of multiple consecutive different lines.
// It then resynchs and looks for further differences.
// Prints up to 10 differences.
//
// I wrote this to do diffs between test results that have different printed
// time results but should be otherwise the same.
// Bracketed time results look like this [Time 2.34]
//
// Warning: this is code written quickly to solve a specific problem -- expect
//          it to be quite messy.
//
// milbo durban dec 05

#if _MSC_VER                        // microsoft
  #define _CRT_SECURE_NO_WARNINGS 1 // disable non-secure function warnings
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/stat.h>

static const int MAX_LEN   = 10000;
static const int MAX_DIFFS = 10;

static void Err(const char *pArgs, ...)    // args like printf
{
    static char s[MAX_LEN];
    va_list pArg;
    va_start(pArg, pArgs);
    vsprintf(s, pArgs, pArg);
    va_end(pArg);
    printf("%s\n", s);
    exit(-1);
}

static char *fgets1(char *s, int n, FILE *stream) // like fgets but discards \r
{
    char *p = fgets(s, n, stream);
    if (p)
    {
        // discard \r
        size_t len = strlen(s);
        if (len >= 2 && s[len-2] == '\r')
        {
            s[len-2] = '\n';
            s[len-1] = 0;
        }
    }
    return p;
}

static void PrintErr(int linenbr,
                     const char path1[], const char line1[],
                     const char path2[], const char line2[])
{
    int linelen = int(strlen(line1))-1;
    int pathlen = int(strlen(path1));
    if (int(strlen(path2)) > pathlen)
        pathlen = int(strlen(path2));
    printf("%*s %5d: %s%s", pathlen, path1, linenbr, line1,
        ((linelen < 0 || line1[linelen] != '\n')? "\n": ""));
    linelen = int(strlen(line2)-1);
    printf("%*s %5d: %s%s\n", pathlen, path2, linenbr, line2,
        ((linelen < 0 || line2[linelen] != '\n')? "\n": ""));
}

// like strcmp but skips (i.e. ignores) text between [square brackets]

static int strcmpskip(char s1[], char s2[])
{
    for (int i = 0, j = 0; s1[i] && s2[j]; i++, j++)
    {
        if (s1[i] == '[')                    // skip []
            while (s1[i] && s1[i] != ']')
                i++;
        if (s2[j] == '[')
            while (s2[j] && s2[j] != ']')    // skip []
                j++;
        if (s1[i] != s2[j])
            return s1[i] - s2[j];
    }
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
        Err("mdiff version 1.3\n\n"
            "usage: mdiff FILE1 FILE2  (FILE2 can be a directory)");

    FILE *file1 = fopen(argv[1], "rt");
    if (NULL == file1)
        Err("mdiff: cannot open %s", argv[1]);

    char path2[MAX_LEN]; strcpy(path2, argv[2]);
    size_t pathlen = strlen(path2);
    if (path2[pathlen-1] == '\\' || path2[pathlen-1] == '/')
        path2[pathlen-1] = 0; // remove trailing slash if any

    // if argv[2] is a directory, create a path by prepending the dir to the basename of argv[1]
#if _MSC_VER // microsoft only to avoid hassles with splitpath
    struct _stat st;
    if (_stat(path2, &st) == 0 && (st.st_mode & S_IFDIR))
    {
        // path2 is a directory
        char base[1024], ext[1024];
        _splitpath(argv[1], NULL, NULL, base, ext);
        _makepath(path2, NULL, argv[2], base, ext);
        for (int i = 0; path2[i]; i++) // convert backslashes (created by makepath) to forward slashes
            if (path2[i] == '\\')
                path2[i] = '/';
    }
#else
    struct stat st;
    if (stat(path2, &st) == 0 && (st.st_mode & S_IFDIR))
        Err("%s is a directory, not supported in this version of mdiff", path2);
#endif
    FILE *file2 = fopen(path2, "rt");
    if (NULL == file2)
        Err("mdiff: cannot open %s", path2);

    // following are static simply to keep these big variables off the stack
    static char line1[MAX_LEN+1], line2[MAX_LEN+1];
    static char prev1[MAX_LEN+1], prev2[MAX_LEN+1];
    int linenbr = 0, err = 0;

    bool prev_has_err = false;

    while (fgets1(line1, MAX_LEN, file1))
    {
        // fLineHasErr prevents us printing the same line twice, if it has multiple errors
        bool line_has_err = false;
        linenbr++;
        if (!fgets1(line2, MAX_LEN, file2))
        {
            if (err++ < MAX_DIFFS)   // can't get line from path2
                PrintErr(linenbr, argv[1], line1, path2, "SHORT FILE");
            err = MAX_DIFFS;        // prevent further messages
            line_has_err = true;
            break;
        }
        if (prev_has_err)
        {
            // basic resync (allows resynch after extra lines in one of the input files)

            if (0 == strcmpskip(line1, prev2))
                fgets1(line1, MAX_LEN, file1);

            if (0 == strcmpskip(line2, prev1))
                fgets1(line2, MAX_LEN, file2);
        }
        int i, j;
        for (i = 0, j = 0; line1[i] && line2[j]; i++, j++)
        {
            if (line1[i] == '[')                      // skip []
                while (line1[i] && line1[i] != ']')
                    i++;
            if (line2[j] == '[')
                while (line2[j] && line2[j] != ']')   // skip []
                    j++;
            if (line1[i] != line2[j])
            {
                if (!prev_has_err            // don't print consecutive differences
                        && err++ < MAX_DIFFS)
                    PrintErr(linenbr, argv[1], line1, path2, line2);
                line_has_err = true;
                prev_has_err = true;
                break;
            }
        }
        if (!line_has_err)
            prev_has_err = false;
        if (!line_has_err && (line1[i] != 0 || line2[j] != 0) // different line lens?
                && err++ < MAX_DIFFS)
        {
            PrintErr(linenbr, argv[1], line1, path2, line2);
            prev_has_err = true;
        }
        if (err >= MAX_DIFFS)
        {
            printf("Reached MAX_DIFFS %d\n", MAX_DIFFS);
            break;
        }
        strcpy(prev1, line1);
        strcpy(prev2, line2);
    }
    if (fgets1(line2, MAX_LEN, file2)) // extra line(s) in File2?
    {
        linenbr++;
        if (err++ < MAX_DIFFS)
            PrintErr(linenbr, argv[1], "SHORT FILE", path2, line2);
    }
    return err != 0;   // return 0 on success
}
