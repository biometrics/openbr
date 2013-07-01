// shapefile.cpp: read and use shape files
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"
#include "opencv/highgui.h"
#include "atface.h"
#include "stasm_regex.h"
#include "shapefile.h"
#include "appmisc.h"

namespace stasm
{
// Return the current line number in the given file.
// This function is very slow but is used only for reporting errors.

static int LineNbr(FILE* file)
{
    CV_Assert(file);
    int n = 0;                       // line number
    long filepos = ftell(file);      // original position in file
    fseek(file, 0, SEEK_SET);        // rewind to start of file
    while (ftell(file) < filepos)    // read lines until back at original posn
    {
        char line[SBIG];
        if (!fgets(line, SBIG-1, file))
            break;                   // should never get here
        n++;
    }
    fseek(file, filepos, SEEK_SET);  // restore original position
    return n;
}

// Like fgets but
//    o no final \n or \r
//    o skips comments (# and anything after it)
//    o skips white space lines or all comment lines
//    o skips space at start of line
//
// TODO Fgets, ReadMatData, ReadMat could be simplified

static const char* Fgets(char* s, int n, FILE* file)
{
    if (s == NULL || n <= 1 || file == NULL) // compat with fgets
        return NULL;
    bool incomment = false, inleadingspace = true;
    int i = 0, c = 0;
    while (1)
    {
        c = fgetc(file);
        if (c == EOF)
        {
            if (i > 0)
                i--;
            break;
        }
        if (c == '\r')
            ;               // ignore CR
        else if (c == '\n')
        {
            if (inleadingspace)
            {
                // start from scratch on the next line in file
                incomment = false;
                i = c = 0;
            }
            else
                break;
        }
        else if (c == '#')
        {
            incomment = true;
            while (i > 0 && (s[i-1] == ' ' || s[i-1] == '\t'))
                i--;        // backspace over preceding blanks
        }
        else if (!incomment)
        {
            if (c != ' ' && c != '\t')
                inleadingspace = false;
            if (!inleadingspace)
            {
                s[i++] = char(c);
                if (i >= n - 1)
                    break;
            }
        }
    }
    s[i] = 0;
    return i == 0? NULL: s;
}

static void Header(         // read and check first line of shape file
    bool&       oldformat,  // out: true if old format shape file
    const char* shapepath,  // in: shape file path
    FILE*       file)       // in
{
    oldformat = false;

    char s[SLEN];
    if (!fgets(s, SLEN-1, file))
        Err("Cannot read %s", shapepath);

    static const char* const whitespace = " \t\n\r";
    char* token = strtok(s, whitespace);
    if (strcmp("ss", token) == 0)
    {
        oldformat = true;
        lprintf("old format shapefile ");
    }
    else if (strcmp("shape", token) != 0)
        Err("%s: bad header (expected \"shape\" or \"ss\")", shapepath);
}

// On return dirs will be a string holding the semi-colon separated list
// of image directories in the given shape file, like "/dir1;/dir2".
//
// On entry, we assume file points to the string, or to comments
// preceding it, which will be ignored.

static void ImgDirs(
    char*       dirs,       // out
    const char* shapepath,  // in: shape file path
    FILE*       file)       // in
{
    char s[SLEN];
    Fgets(s, SLEN-1, file); // will skip blank lines and comments, if any

    static const char* const whitespace = " \t\n\r";
    char* token = strtok(s, whitespace);
    if (!token || 0 != strcmp(token, "Directories"))
        Err("Expected \"Directories\" in line %d of %s",
            LineNbr(file), shapepath);

    token = strtok(NULL, whitespace);
    if (!token)
        Err("Cannot read image directories in line %d of %s",
            LineNbr(file), shapepath);

    strncpy_(dirs, token, SLEN);
    ConvertBackslashesToForwardAndStripFinalSlash(dirs);
}

static void PrematureEndOfFile(FILE* file, const char* path)
{
    long n = LineNbr(file);
    if (n > 0)
        Err("%s(%d): premature end of file", path, n);
    else
        Err("Cannot read from %s", path);
}

static void SkipToEndOfLine(FILE* file, const char* path)
{
    int c = ' ';
    while (c != '\n' && c != EOF)
        c = fgetc(file);
    if (c == EOF)
        PrematureEndOfFile(file, path);
}

// Read the data fields of a matrix
// This assumes we have already read the header "{ nrows ncols".
// It reads up to but not including the final "}".
//
// Known issue: comments not at the start of a line must be preceded by a space.

static void ReadMatData(
    Shape&      mat,    // out
    int         nrows,   // in
    int         ncols,   // in
    FILE*       file,    // in
    const char* path)    // in: for error reporting
{
    double* data = Buf(mat);
    for (int i = 0; i < ncols * nrows; i++)
    {
        // skip comments and white space
        int c = ' ';
        while (c == ' ' || c == '\t' || c == '\n' || c == '\r')
        {
            c = fgetc(file);
            if (c == '#') // comment
            {
                SkipToEndOfLine(file, path);
                c = fgetc(file);
            }
        }
        if (c == EOF)
            PrematureEndOfFile(file, path);
        else
        {
            ungetc(c, file);
            float temp; // microsoft compiler can't sscanf doubles so use float
            if (!fscanf(file, "%g", &temp))
                Err("%s(%d): Cannot read %dx%d matrix",
                    path, LineNbr(file), nrows, ncols);
            data[i] = temp;
        }
    }
}

static bool ReadMat(  // true if read the mat, false if no (more) mats in file
    char*       base, // out: basename in tag
    unsigned&   bits, // out: hex bits in tag
    Shape&      mat,  // out: the matrix
    FILE*       file, // in:  pointer to the shape file
    const char* path) // in:  for error messages
{
    char s[SLEN];     // the string tag before the matrix
    while (1)
    {
        int c = fgetc(file);
        if (c == EOF)
            return false;   // note return
        if (c == '{')
            break;          // note break
        else if (c == '#')
            SkipToEndOfLine(file, path);
        else if (c == '\n' || c == '\r' || c == '\t' || c == ' ') // white space
            ;
        else if (c == '"') // old format tag (enclosed in quotes)
            ;
        else    // any other char, assume it is the start of the tag
        {
            s[0] = char(c);
            if (!Fgets(s+1, SLEN-1, file))
                Err("%s(%d): Read failed (premature EOF)",
                    path, LineNbr(file));
            // remove trailing white space and final quote if any
            int i = STRNLEN(s, SLEN) - 1;
            CV_Assert(i >= 4);
            while (s[i] == ' ' || s[i] == '\t' || s[i] == '"')
                i--;
            s[i+1] = 0;
        }
    }
    if (!s[0])
        Err("%s(%d): Empty tag", path, LineNbr(file));
    if (s[4] != ' ' && s[8] != ' ') // hex string must be 4 or 8 chars
        Err("%s(%d): Malformed tag", path, LineNbr(file));
    if (2 != sscanf(s, "%x %s", &bits, base))
        Err("%s(%d): Malformed tag", path, LineNbr(file));

    int nrows, ncols; int c;
    if (2 != fscanf(file, "%d %d", &nrows, &ncols))
        Err("%s(%d): Cannot read matrix size", path, LineNbr(file));
    if (ncols < 1 || nrows > MAX_MAT_DIM)
        Err("%s(%d): Invalid number of rows %d", path, LineNbr(file), nrows);
    if (ncols < 1 || ncols > MAX_MAT_DIM)
        Err("%s(%d): Invalid number of columns %d", path, LineNbr(file), ncols);

    mat.create(nrows, ncols);
    ReadMatData(mat, nrows, ncols, file, path);

    // make sure that next non-white char is matrix terminator '}'

    c = ' ';
    while (c == ' ' || c == '\t' || c == '\n' || c == '\r') // skip white space
        if (EOF == (c = fgetc(file))) // assignment is intentional
            Err("%s(%d): Cannot read matrix\n"
                "       Reached EOF before finding \"}\"",
                path, LineNbr(file));
    if (c == '#')
        Err("%s(%d): Comment not allowed here", path, LineNbr(file));
    if (c != '}')
        Err("%s(%d): Footer is not \"}\" "
            "(too many or two few entries in matrix?)", path, LineNbr(file));

    return true; // success
}

void ShapeFile::Open_(     // read shape file from disk
    const char* shapepath) // in
{
    lprintf("Reading %s: ", shapepath);
    STRCPY(shapepath_, shapepath);
    FILE* file = fopen(shapepath, "rb");
    if (!file)
        Err("Cannot open %s", shapepath);
    bool oldformat = false;
    Header(oldformat, shapepath, file);
    ImgDirs(dirs_, shapepath, file);
    shapes_.clear();
    bases_.clear();
    bits_.clear();
    poses_.clear();
    char base[SLEN]; unsigned bits; Shape shape;
    nchar_ = 0;
    int nrows = -1;
    while (ReadMat(base, bits, shape, file, shapepath))
    {
        bool normalshape = true;
        if (oldformat)
        {
            if (bits & FA_Meta)  // metashape?
                normalshape = false;
        }
        else if (bits & AT_Meta) // metashape?
        {
            normalshape = false;
            if (bits == AT_Pose)
            {
                CV_Assert(shape.rows == 1 && shape.cols == 4);
                poses_[base] = shape.clone();
            }
        }
        if (normalshape)
        {
            // check that all shapes have same number of points
            if (nrows == -1) // first shape?
                nrows = shape.rows;
            else if (shape.rows != nrows)
                Err("%s has %d row%s but %s has %d row%s",
                    base, shape.rows, plural(shape.rows),
                    bases_[0].c_str(), nrows, plural(nrows));
            shapes_.push_back(shape.clone());
            bases_.push_back(base);
            bits_.push_back(bits);
            int len = STRNLEN(base, 100);
            if (len > nchar_)
                nchar_ = len;
        }
    }
    fclose(file);
    nshapes_ = NSIZE(shapes_);
    lprintf("%d shape%s\n", nshapes_, plural(nshapes_));
    if (nshapes_ == 0)
        Err("No shapes in %s", shapepath);
}

void ShapeFile::SubsetRegex_( // select shapes matching regex
    const char* sregex)       // in: regex string, NULL or "" to match any
{
    if (sregex && sregex[0])
    {
        const regex re(CompileRegex(sregex));
        int j = 0;
        for (int i = 0; i < nshapes_; i++)
            if (MatchRegex(bases_[i], re))
                {
                    shapes_[j] = shapes_[i];
                    bases_ [j] = bases_ [i];
                    bits_  [j] = bits_  [i];
                    j++;
                }

        if (j == 0)
            Err("No shapes in %s match \"%s\"", shapepath_, sregex);
        shapes_.resize(j);
        nshapes_ = j;
    }
}

static void RandInts(
    vec_int& ints,    // out: scrambled integers in range 0 ... NSIZE(ints)-1
    int      seed)    // in:  random seed
{
    const int n = NSIZE(ints);
    CV_Assert(n > 0);
    if (n > RAND_MAX)
        Err("vector size %d is too big (max allowed is %d)", n, RAND_MAX);
    CV_Assert(seed != 0);
    if (seed == 1)       // 1 has a special meaning which we don't want
        seed = int(1e6); // arb

    int i;
    for (i = 0; i < n; i++)
        ints[i] = i;

    srand(seed);

    // We use our own random shuffle here because different compilers
    // give different results which messes up regression testing.
    // (I think only Visual C 6.0 is incompatible with everyone else?)
    //
    // Following code is equivalent to
    //    random_shuffle(ints.begin(), ints.end(),
    //       pointer_to_unary_function<int,int>(RandInt));

    vec_int::iterator it = ints.begin();
    for (i = 2; ++it != ints.end(); i++)
        iter_swap(it, ints.begin() + rand() % n);
}

static void CheckNshapes(
    int         nshapes,      // in
    int         nallowed,     // in
    const char* sregex,       // in: regex string (used only for err msgs)
    const char* shapepath)    // in: shapfile path (used only for err msgs)
{
    if (nshapes <= 0)
        Err("Invalid number of shapes %d", nshapes);
    if (nshapes > nallowed)
    {
        char s1[SLEN]; s1[0] = 0;
        if (sregex && sregex[0]) // regular expression specified?
            sprintf(s1, "after matching regex %s", sregex);
        else
            sprintf(s1, "in %s", shapepath);
        Err("Want %d shape%s but only %d shape%s %s",
             nshapes, plural(nshapes),
             nallowed, plural(nallowed), s1);
    }
}

void ShapeFile::SubsetN_(     // select nshapes
    int nshapes,              // in: number of shapes to select
    int seed,                 // in: if 0 use first nshapes,
                              //     else rand subset of nshapes
    const char* sregex)       // in: regex string (used only for err msgs)
{
    CheckNshapes(nshapes, nshapes_, sregex, shapepath_);
    if (seed && nshapes > 1)
    {
        vec_int ints(nshapes_);
        RandInts(ints, seed);
        ints.resize(nshapes);
        sort(ints.begin(), ints.end());
        for (int i = 0; i < nshapes; i++)
        {
            const int j = ints[i];
            shapes_[i] = shapes_[j];
            bases_[i] = bases_[j];
            bits_[i] = bits_[j];
        }
    }
    nshapes_ = nshapes;
    shapes_.resize(nshapes);
    bases_.resize(nshapes);
    bits_.resize(nshapes);
}

void ShapeFile::Subset_( // select a subset of the shapes
    int         nshapes, // in: number of shapes to select
    int         seed,    // in: if 0 use first nshapes,
                         //     else rand subset of nshapes
    const char* sregex)  // in: regex string, NULL or "" to match any
{
    SubsetRegex_(sregex);                // select shapes matching the regex

    if (nshapes)
        SubsetN_(nshapes, seed, sregex); // select nshapes

    nchar_ = 0; // nbr of chars in longest string in bases_
    for (int i = 0; i < NSIZE(bases_); i++)
        if (int(bases_[i].length()) > nchar_)
            nchar_ = int(bases_[i].length());

    if (nshapes == 1 || nshapes_ == 1)
        lprintf("Using 1 shape");
    else if (nshapes == 0)
        lprintf("Using all %d shapes", nshapes_);
    else if (seed == 0)
        lprintf("Using the first %d shape%s", nshapes, plural(nshapes));
    else
        lprintf("Using a random subset of %d shape%s", nshapes, plural(nshapes));
    if (sregex && sregex[0])
        lprintf(" matching %s", sregex);
    if (NSIZE(bases_) > 1)
        lprintf(" (%s ... %s)",
                bases_[0].c_str(), bases_[NSIZE(bases_)-1].c_str());
    lprintf("\n");
}

} // namespace stasm
