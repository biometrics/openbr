/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_scale_div.h"
#include "qwt_math.h"
#include "qwt_interval.h"
#include <qalgorithms.h>

//! Construct an invalid QwtScaleDiv instance.
QwtScaleDiv::QwtScaleDiv():
    d_lowerBound( 0.0 ),
    d_upperBound( 0.0 ),
    d_isValid( false )
{
}

/*!
  Construct QwtScaleDiv instance.

  \param interval Interval
  \param ticks List of major, medium and minor ticks
*/
QwtScaleDiv::QwtScaleDiv( const QwtInterval &interval,
        QList<double> ticks[NTickTypes] ):
    d_lowerBound( interval.minValue() ),
    d_upperBound( interval.maxValue() ),
    d_isValid( true )
{
    for ( int i = 0; i < NTickTypes; i++ )
        d_ticks[i] = ticks[i];
}

/*!
  Construct QwtScaleDiv instance.

  \param lowerBound First interval limit
  \param upperBound Second interval limit
  \param ticks List of major, medium and minor ticks
*/
QwtScaleDiv::QwtScaleDiv(
        double lowerBound, double upperBound,
        QList<double> ticks[NTickTypes] ):
    d_lowerBound( lowerBound ),
    d_upperBound( upperBound ),
    d_isValid( true )
{
    for ( int i = 0; i < NTickTypes; i++ )
        d_ticks[i] = ticks[i];
}

/*!
   Change the interval
   \param interval Interval
*/
void QwtScaleDiv::setInterval( const QwtInterval &interval )
{
    setInterval( interval.minValue(), interval.maxValue() );
}

/*!
  \brief Equality operator
  \return true if this instance is equal to other
*/
bool QwtScaleDiv::operator==( const QwtScaleDiv &other ) const
{
    if ( d_lowerBound != other.d_lowerBound ||
        d_upperBound != other.d_upperBound ||
        d_isValid != other.d_isValid )
    {
        return false;
    }

    for ( int i = 0; i < NTickTypes; i++ )
    {
        if ( d_ticks[i] != other.d_ticks[i] )
            return false;
    }

    return true;
}

/*!
  \brief Inequality
  \return true if this instance is not equal to s
*/
bool QwtScaleDiv::operator!=( const QwtScaleDiv &s ) const
{
    return ( !( *this == s ) );
}

//! Invalidate the scale division
void QwtScaleDiv::invalidate()
{
    d_isValid = false;

    // detach arrays
    for ( int i = 0; i < NTickTypes; i++ )
        d_ticks[i].clear();

    d_lowerBound = d_upperBound = 0;
}

//! Check if the scale division is valid
bool QwtScaleDiv::isValid() const
{
    return d_isValid;
}

/*!
  Return if a value is between lowerBound() and upperBound()

  \param value Value
  \return true/false
*/
bool QwtScaleDiv::contains( double value ) const
{
    if ( !d_isValid )
        return false;

    const double min = qMin( d_lowerBound, d_upperBound );
    const double max = qMax( d_lowerBound, d_upperBound );

    return value >= min && value <= max;
}

//! Invert the scale divison
void QwtScaleDiv::invert()
{
    qSwap( d_lowerBound, d_upperBound );

    for ( int i = 0; i < NTickTypes; i++ )
    {
        QList<double>& ticks = d_ticks[i];

        const int size = ticks.count();
        const int size2 = size / 2;

        for ( int i = 0; i < size2; i++ )
            qSwap( ticks[i], ticks[size - 1 - i] );
    }
}

/*!
    Assign ticks

   \param type MinorTick, MediumTick or MajorTick
   \param ticks Values of the tick positions
*/
void QwtScaleDiv::setTicks( int type, const QList<double> &ticks )
{
    if ( type >= 0 || type < NTickTypes )
        d_ticks[type] = ticks;
}

/*!
   Return a list of ticks

   \param type MinorTick, MediumTick or MajorTick
*/
const QList<double> &QwtScaleDiv::ticks( int type ) const
{
    if ( type >= 0 || type < NTickTypes )
        return d_ticks[type];

    static QList<double> noTicks;
    return noTicks;
}
