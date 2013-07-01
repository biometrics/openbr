// appmisc.h: miscellaneous defs for apps but not needed by the Stasm library itself
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_APPMISC_H
#define STASM_APPMISC_H

namespace stasm
{
inline const char* plural(int n)
{
    return n == 1? "": "s";
}

class Pacifier // print pacifier when i is at every 10% of n
{
public:
    Pacifier(int n, int nmin=50) // constructor
    {
        n_ = n;
        nmin_ = nmin; // n must be at least nmin to print anything
        boundary_ = 0;
    }
    void Print_(int i) // print pacifier if i has reached next 10% boundary
    {
        CV_Assert(i >= 0 && i <= n_);
        if (n_ > nmin_ && i >= boundary_)
        {
            const int n10 = ((n_ + 10) / 10) * 10;
            printf("%d", (boundary_ * 10) / n10);
            fflush(stdout);
            boundary_ += n10 / 10;
        }
    }
    void End_(void) // print final 0
    {
        if (n_ > nmin_)
        {
            printf("0");
            fflush(stdout);
        }
    }
private:
    int n_, nmin_, boundary_;
    DISALLOW_COPY_AND_ASSIGN(Pacifier);
};

const Shape LandmarksAsShape(
    const float* landmarks);    // in

void Fprintf(                   // like fprintf but issue err if can't write
    FILE*       file,           // in
    const char* format, ...);   // in

void LogShapeAsCsv(             // print shape in CSV format to log file
    const MAT&  mat,            // in
    const char* path);          // in

void CropCimgToShapeWithMargin( // crop the image so the shape fills the image
    CImage& img,                // io
    const  Shape& Shape,        // in
    double xmargin=-1,          // in: -1 (default) means auto choose margin
    double ymargin=-1);         // in: -1 (default) means auto choose margin

char* PathGivenDirs(
    const char* base,           // in
    const char* dirs,           // in: dir names separated by semicolons
    const char* shapepath);     // in: path of shape file holding dirs, for err msgs

} // namespace stasm
#endif // STASM_APPMISC_H
