/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_MATH_H
#define QWT_MATH_H

#include "qwt_global.h"

#if defined(_MSC_VER)
/*
  Microsoft says:

  Define _USE_MATH_DEFINES before including math.h to expose these macro
  definitions for common math constants.  These are placed under an #ifdef
  since these commonly-defined names are not part of the C/C++ standards.
*/
#define _USE_MATH_DEFINES 1
#endif

#include <qpoint.h>
#include <qmath.h>
#include "qwt_global.h"

#ifndef LOG10_2
#define LOG10_2     0.30102999566398119802  /* log10(2) */
#endif

#ifndef LOG10_3
#define LOG10_3     0.47712125471966243540  /* log10(3) */
#endif

#ifndef LOG10_5
#define LOG10_5     0.69897000433601885749  /* log10(5) */
#endif

#ifndef M_2PI
#define M_2PI       6.28318530717958623200  /* 2 pi */
#endif

#ifndef LOG_MIN
//! Mininum value for logarithmic scales
#define LOG_MIN 1.0e-100
#endif

#ifndef LOG_MAX
//! Maximum value for logarithmic scales
#define LOG_MAX 1.0e100
#endif

#ifndef M_E
#define M_E            2.7182818284590452354   /* e */
#endif

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074   /* log_2 e */
#endif

#ifndef M_LOG10E
#define M_LOG10E    0.43429448190325182765  /* log_10 e */
#endif

#ifndef M_LN2
#define M_LN2       0.69314718055994530942  /* log_e 2 */
#endif

#ifndef M_LN10
#define M_LN10         2.30258509299404568402  /* log_e 10 */
#endif

#ifndef M_PI
#define M_PI        3.14159265358979323846  /* pi */
#endif

#ifndef M_PI_2
#define M_PI_2      1.57079632679489661923  /* pi/2 */
#endif

#ifndef M_PI_4
#define M_PI_4      0.78539816339744830962  /* pi/4 */
#endif

#ifndef M_1_PI
#define M_1_PI      0.31830988618379067154  /* 1/pi */
#endif

#ifndef M_2_PI
#define M_2_PI      0.63661977236758134308  /* 2/pi */
#endif

#ifndef M_2_SQRTPI
#define M_2_SQRTPI  1.12837916709551257390  /* 2/sqrt(pi) */
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880  /* sqrt(2) */
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2   0.70710678118654752440  /* 1/sqrt(2) */
#endif

QWT_EXPORT double qwtGetMin( const double *array, int size );
QWT_EXPORT double qwtGetMax( const double *array, int size );

/*!
  \brief Compare 2 values, relative to an interval

  Values are "equal", when :
  \f$\cdot value2 - value1 <= abs(intervalSize * 10e^{-6})\f$

  \param value1 First value to compare
  \param value2 Second value to compare
  \param intervalSize interval size

  \return 0: if equal, -1: if value2 > value1, 1: if value1 > value2
*/
inline int qwtFuzzyCompare( double value1, double value2, double intervalSize )
{
    const double eps = qAbs( 1.0e-6 * intervalSize );

    if ( value2 - value1 > eps )
        return -1;

    if ( value1 - value2 > eps )
        return 1;

    return 0;
}


inline bool qwtFuzzyGreaterOrEqual( double d1, double d2 )
{
    return ( d1 >= d2 ) || qFuzzyCompare( d1, d2 );
}

inline bool qwtFuzzyLessOrEqual( double d1, double d2 )
{
    return ( d1 <= d2 ) || qFuzzyCompare( d1, d2 );
}

//! Return the sign
inline int qwtSign( double x )
{
    if ( x > 0.0 )
        return 1;
    else if ( x < 0.0 )
        return ( -1 );
    else
        return 0;
}

//! Return the square of a number
inline double qwtSqr( double x )
{
    return x * x;
}

//! Like qRound, but without converting the result to an int
inline double qwtRoundF(double d)
{ 
    return ::floor( d + 0.5 );
}

//! Like qFloor, but without converting the result to an int
inline double qwtFloorF(double d)
{ 
    return ::floor( d );
}

//! Like qCeil, but without converting the result to an int
inline double qwtCeilF(double d)
{ 
    return ::ceil( d );
}

#endif
