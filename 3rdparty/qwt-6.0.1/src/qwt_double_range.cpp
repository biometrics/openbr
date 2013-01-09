/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_double_range.h"
#include "qwt_math.h"

#if QT_VERSION < 0x040601
#define qFabs(x) ::fabs(x)
#endif

class QwtDoubleRange::PrivateData
{       
public: 
    PrivateData():
        minValue( 0.0 ),
        maxValue( 0.0 ),
        step( 1.0 ),
        pageSize( 1 ),
        isValid( false ),
        value( 0.0 ),
        exactValue( 0.0 ),
        exactPrevValue( 0.0 ),
        prevValue( 0.0 ),
        periodic( false )
    {
    }

    double minValue;
    double maxValue;
    double step;
    int pageSize;

    bool isValid;
    double value;
    double exactValue;
    double exactPrevValue;
    double prevValue;

    bool periodic;
};

/*!
  The range is initialized to [0.0, 100.0], the
  step size to 1.0, and the value to 0.0.
*/
QwtDoubleRange::QwtDoubleRange()
{
    d_data = new PrivateData();
}

//! Destroys the QwtDoubleRange
QwtDoubleRange::~QwtDoubleRange()
{
    delete d_data;
}

//! Set the value to be valid/invalid
void QwtDoubleRange::setValid( bool isValid )
{
    if ( isValid != d_data->isValid )
    {
        d_data->isValid = isValid;
        valueChange();
    }
}

//! Indicates if the value is valid
bool QwtDoubleRange::isValid() const
{
    return d_data->isValid;
}

void QwtDoubleRange::setNewValue( double value, bool align )
{
    d_data->prevValue = d_data->value;

    const double vmin = qMin( d_data->minValue, d_data->maxValue );
    const double vmax = qMax( d_data->minValue, d_data->maxValue );

    if ( value < vmin )
    {
        if ( d_data->periodic && vmin != vmax )
        {
            d_data->value = value + 
                qwtCeilF( ( vmin - value ) / ( vmax - vmin ) ) * ( vmax - vmin );
        }
        else
            d_data->value = vmin;
    }
    else if ( value > vmax )
    {
        if ( ( d_data->periodic ) && ( vmin != vmax ) )
        {
            d_data->value = value - 
                qwtCeilF( ( value - vmax ) / ( vmax - vmin ) ) * ( vmax - vmin );
        }
        else
            d_data->value = vmax;
    }
    else
    {
        d_data->value = value;
    }

    d_data->exactPrevValue = d_data->exactValue;
    d_data->exactValue = d_data->value;

    if ( align )
    {
        if ( d_data->step != 0.0 )
        {
            d_data->value = d_data->minValue +
                qRound( ( d_data->value - d_data->minValue ) / d_data->step ) * d_data->step;
        }
        else
            d_data->value = d_data->minValue;

        const double minEps = 1.0e-10;
        // correct rounding error at the border
        if ( qFabs( d_data->value - d_data->maxValue ) < minEps * qAbs( d_data->step ) )
            d_data->value = d_data->maxValue;

        // correct rounding error if value = 0
        if ( qFabs( d_data->value ) < minEps * qAbs( d_data->step ) )
            d_data->value = 0.0;
    }

    if ( !d_data->isValid || d_data->prevValue != d_data->value )
    {
        d_data->isValid = true;
        valueChange();
    }
}

/*!
  \brief  Adjust the value to the closest point in the step raster.
  \param x value
  \warning The value is clipped when it lies outside the range.
  When the range is QwtDoubleRange::periodic, it will
  be mapped to a point in the interval such that
  \verbatim new value := x + n * (max. value - min. value)\endverbatim
  with an integer number n.
*/
void QwtDoubleRange::fitValue( double x )
{
    setNewValue( x, true );
}


/*!
  \brief Set a new value without adjusting to the step raster
  \param x new value
  \warning The value is clipped when it lies outside the range.
  When the range is QwtDoubleRange::periodic, it will
  be mapped to a point in the interval such that
  \verbatim new value := x + n * (max. value - min. value)\endverbatim
  with an integer number n.
*/
void QwtDoubleRange::setValue( double x )
{
    setNewValue( x, false );
}

/*!
  \brief Specify  range and step size

  \param vmin   lower boundary of the interval
  \param vmax   higher boundary of the interval
  \param vstep  step width
  \param pageSize  page size in steps
  \warning
  \li A change of the range changes the value if it lies outside the
      new range. The current value
      will *not* be adjusted to the new step raster.
  \li vmax < vmin is allowed.
  \li If the step size is left out or set to zero, it will be
      set to 1/100 of the interval length.
  \li If the step size has an absurd value, it will be corrected
      to a better one.
*/
void QwtDoubleRange::setRange( 
    double vmin, double vmax, double vstep, int pageSize )
{
    const bool rchg = ( d_data->maxValue != vmax || d_data->minValue != vmin );

    if ( rchg )
    {
        d_data->minValue = vmin;
        d_data->maxValue = vmax;
    }

    // look if the step width has an acceptable
    // value or otherwise change it.
    setStep( vstep );

    // limit page size
    const int max = 
        int( qAbs( ( d_data->maxValue - d_data->minValue ) / d_data->step ) );
    d_data->pageSize = qBound( 0, pageSize, max );

    // If the value lies out of the range, it
    // will be changed. Note that it will not be adjusted to
    // the new step width.
    setNewValue( d_data->value, false );

    // call notifier after the step width has been
    // adjusted.
    if ( rchg )
        rangeChange();
}

/*!
  \brief Change the step raster
  \param vstep new step width
  \warning The value will \e not be adjusted to the new step raster.
*/
void QwtDoubleRange::setStep( double vstep )
{
    const double intv = d_data->maxValue - d_data->minValue;

    double newStep;
    if ( vstep == 0.0 )
    {
        const double defaultRelStep = 1.0e-2;
        newStep = intv * defaultRelStep;
    }
    else
    {
        if ( ( intv > 0.0 && vstep < 0.0 ) || ( intv < 0.0 && vstep > 0.0 ) )
            newStep = -vstep;
        else
            newStep = vstep;

        const double minRelStep = 1.0e-10;
        if ( qFabs( newStep ) < qFabs( minRelStep * intv ) )
            newStep = minRelStep * intv;
    }

    if ( newStep != d_data->step )
    {
        d_data->step = newStep;
        stepChange();
    }
}


/*!
  \brief Make the range periodic

  When the range is periodic, the value will be set to a point
  inside the interval such that

  \verbatim point = value + n * width \endverbatim

  if the user tries to set a new value which is outside the range.
  If the range is nonperiodic (the default), values outside the
  range will be clipped.

  \param tf true for a periodic range
*/
void QwtDoubleRange::setPeriodic( bool tf )
{
    d_data->periodic = tf;
}

/*!
  \brief Increment the value by a specified number of steps
  \param nSteps Number of steps to increment
  \warning As a result of this operation, the new value will always be
       adjusted to the step raster.
*/
void QwtDoubleRange::incValue( int nSteps )
{
    if ( isValid() )
        setNewValue( d_data->value + double( nSteps ) * d_data->step, true );
}

/*!
  \brief Increment the value by a specified number of pages
  \param nPages Number of pages to increment.
        A negative number decrements the value.
  \warning The Page size is specified in the constructor.
*/
void QwtDoubleRange::incPages( int nPages )
{
    if ( isValid() )
    {
        const double off = d_data->step * d_data->pageSize * nPages; 
        setNewValue( d_data->value + off, true );
    }
}

/*!
  \brief Notify a change of value

  This virtual function is called whenever the value changes.
  The default implementation does nothing.
*/
void QwtDoubleRange::valueChange()
{
}


/*!
  \brief Notify a change of the range

  This virtual function is called whenever the range changes.
  The default implementation does nothing.
*/
void QwtDoubleRange::rangeChange()
{
}


/*!
  \brief Notify a change of the step size

  This virtual function is called whenever the step size changes.
  The default implementation does nothing.
*/
void QwtDoubleRange::stepChange()
{
}

/*!
  \return the step size
  \sa setStep(), setRange()
*/
double QwtDoubleRange::step() const
{
    return qAbs( d_data->step );
}

/*!
  \brief Returns the value of the second border of the range

  maxValue returns the value which has been specified
  as the second parameter in  QwtDoubleRange::setRange.

  \sa setRange()
*/
double QwtDoubleRange::maxValue() const
{
    return d_data->maxValue;
}

/*!
  \brief Returns the value at the first border of the range

  minValue returns the value which has been specified
  as the first parameter in  setRange().

  \sa setRange()
*/
double QwtDoubleRange::minValue() const
{
    return d_data->minValue;
}

/*!
  \brief Returns true if the range is periodic
  \sa setPeriodic()
*/
bool QwtDoubleRange::periodic() const
{
    return d_data->periodic;
}

//! Returns the page size in steps.
int QwtDoubleRange::pageSize() const
{
    return d_data->pageSize;
}

//! Returns the current value.
double QwtDoubleRange::value() const
{
    return d_data->value;
}

/*!
  \brief Returns the exact value

  The exact value is the value which QwtDoubleRange::value would return
  if the value were not adjusted to the step raster. It differs from
  the current value only if fitValue() or incValue() have been used before. 
  This function is intended for internal use in derived classes.
*/
double QwtDoubleRange::exactValue() const
{
    return d_data->exactValue;
}

//! Returns the exact previous value
double QwtDoubleRange::exactPrevValue() const
{
    return d_data->exactPrevValue;
}

//! Returns the previous value
double QwtDoubleRange::prevValue() const
{
    return d_data->prevValue;
}
