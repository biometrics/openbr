/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_scale_map.h"
#include <qrect.h>
#include <qalgorithms.h>
#include <qmath.h>
#include <qdebug.h>

#if QT_VERSION < 0x040601
#define qExp(x) ::exp(x)
#endif

//! Smallest allowed value for logarithmic scales: 1.0e-150
QT_STATIC_CONST_IMPL double QwtScaleMap::LogMin = 1.0e-150;

//! Largest allowed value for logarithmic scales: 1.0e150
QT_STATIC_CONST_IMPL double QwtScaleMap::LogMax = 1.0e150;

//! Constructor for a linear transformation
QwtScaleTransformation::QwtScaleTransformation( Type type ):
    d_type( type )
{
}

//! Destructor
QwtScaleTransformation::~QwtScaleTransformation()
{
}

//! Create a clone of the transformation
QwtScaleTransformation *QwtScaleTransformation::copy() const
{
    return new QwtScaleTransformation( d_type );
}

/*!
  \brief Transform a value from the coordinate system of a scale
         into the coordinate system of the paint device

  \param s  Value related to the coordinate system of the scale
  \param s1 First border of the coordinate system of the scale
  \param s2 Second border of the coordinate system of the scale
  \param p1 First border of the coordinate system of the paint device
  \param p2 Second border of the coordinate system of the paint device
  \return
  <dl>
  <dt>linear mapping:<dd>p1 + (p2 - p1) / (s2 - s1) * (s - s1);</dd>
  </dl>
  <dl>
  <dt>log10 mapping: <dd>p1 + (p2 - p1) / log(s2 / s1) * log(s / s1);</dd>
  </dl>
*/

double QwtScaleTransformation::xForm(
    double s, double s1, double s2, double p1, double p2 ) const
{
    if ( d_type == Log10 )
        return p1 + ( p2 - p1 ) / log( s2 / s1 ) * log( s / s1 );
    else
        return p1 + ( p2 - p1 ) / ( s2 - s1 ) * ( s - s1 );
}

/*!
  \brief Transform a value from the coordinate system of the paint device
         into the coordinate system of a scale.

  \param p Value related to the coordinate system of the paint device 
  \param p1 First border of the coordinate system of the paint device
  \param p2 Second border of the coordinate system of the paint device
  \param s1 First border of the coordinate system of the scale
  \param s2 Second border of the coordinate system of the scale
  \return
  <dl>
  <dt>linear mapping:<dd>s1 + ( s2 - s1 ) / ( p2 - p1 ) * ( p - p1 );</dd>
  </dl>
  <dl>
  <dt>log10 mapping:<dd>exp((p - p1) / (p2 - p1) * log(s2 / s1)) * s1;</dd>
  </dl>
*/

double QwtScaleTransformation::invXForm( double p, double p1, double p2,
    double s1, double s2 ) const
{
    if ( d_type == Log10 )
        return qExp( ( p - p1 ) / ( p2 - p1 ) * log( s2 / s1 ) ) * s1;
    else
        return s1 + ( s2 - s1 ) / ( p2 - p1 ) * ( p - p1 );
}

/*!
  \brief Constructor

  The scale and paint device intervals are both set to [0,1].
*/
QwtScaleMap::QwtScaleMap():
    d_s1( 0.0 ),
    d_s2( 1.0 ),
    d_p1( 0.0 ),
    d_p2( 1.0 ),
    d_cnv( 1.0 )
{
    d_transformation = new QwtScaleTransformation(
        QwtScaleTransformation::Linear );
}

//! Copy constructor
QwtScaleMap::QwtScaleMap( const QwtScaleMap& other ):
    d_s1( other.d_s1 ),
    d_s2( other.d_s2 ),
    d_p1( other.d_p1 ),
    d_p2( other.d_p2 ),
    d_cnv( other.d_cnv )
{
    d_transformation = other.d_transformation->copy();
}

/*!
  Destructor
*/
QwtScaleMap::~QwtScaleMap()
{
    delete d_transformation;
}

//! Assignment operator
QwtScaleMap &QwtScaleMap::operator=( const QwtScaleMap & other )
{
    d_s1 = other.d_s1;
    d_s2 = other.d_s2;
    d_p1 = other.d_p1;
    d_p2 = other.d_p2;
    d_cnv = other.d_cnv;

    delete d_transformation;
    d_transformation = other.d_transformation->copy();

    return *this;
}

/*!
   Initialize the map with a transformation
*/
void QwtScaleMap::setTransformation(
    QwtScaleTransformation *transformation )
{
    if ( transformation == NULL )
        return;

    if ( transformation != d_transformation )
    {
        delete d_transformation;
        d_transformation = transformation;
    }

    setScaleInterval( d_s1, d_s2 );
}

//! Get the transformation
const QwtScaleTransformation *QwtScaleMap::transformation() const
{
    return d_transformation;
}

/*!
  \brief Specify the borders of the scale interval
  \param s1 first border
  \param s2 second border
  \warning logarithmic scales might be aligned to [LogMin, LogMax]
*/
void QwtScaleMap::setScaleInterval( double s1, double s2 )
{
    if ( d_transformation->type() == QwtScaleTransformation::Log10 )
    {
        if ( s1 < LogMin )
            s1 = LogMin;
        else if ( s1 > LogMax )
            s1 = LogMax;

        if ( s2 < LogMin )
            s2 = LogMin;
        else if ( s2 > LogMax )
            s2 = LogMax;
    }

    d_s1 = s1;
    d_s2 = s2;

    if ( d_transformation->type() != QwtScaleTransformation::Other )
        newFactor();
}

/*!
  \brief Specify the borders of the paint device interval
  \param p1 first border
  \param p2 second border
*/
void QwtScaleMap::setPaintInterval( double p1, double p2 )
{
    d_p1 = p1;
    d_p2 = p2;

    if ( d_transformation->type() != QwtScaleTransformation::Other )
        newFactor();
}

/*!
  \brief Re-calculate the conversion factor.
*/
void QwtScaleMap::newFactor()
{
    d_cnv = 0.0;

    switch ( d_transformation->type() )
    {
        case QwtScaleTransformation::Linear:
        {
            if ( d_s2 != d_s1 )
                d_cnv = ( d_p2 - d_p1 ) / ( d_s2 - d_s1 );
            break;
        }
        case QwtScaleTransformation::Log10:
        {
            if ( d_s1 != 0 )
                d_cnv = ( d_p2 - d_p1 ) / log( d_s2 / d_s1 );
            break;
        }
        default:;
    }
}

/*!
   Transform a rectangle from scale to paint coordinates

   \param xMap X map
   \param yMap Y map
   \param rect Rectangle in scale coordinates
   \return Rectangle in paint coordinates

   \sa invTransform()
*/
QRectF QwtScaleMap::transform( const QwtScaleMap &xMap,
    const QwtScaleMap &yMap, const QRectF &rect )
{
    double x1 = xMap.transform( rect.left() );
    double x2 = xMap.transform( rect.right() );
    double y1 = yMap.transform( rect.top() );
    double y2 = yMap.transform( rect.bottom() );

    if ( x2 < x1 )
        qSwap( x1, x2 );
    if ( y2 < y1 )
        qSwap( y1, y2 );

    if ( qwtFuzzyCompare( x1, 0.0, x2 - x1 ) == 0 )
        x1 = 0.0;
    if ( qwtFuzzyCompare( x2, 0.0, x2 - x1 ) == 0 )
        x2 = 0.0;
    if ( qwtFuzzyCompare( y1, 0.0, y2 - y1 ) == 0 )
        y1 = 0.0;
    if ( qwtFuzzyCompare( y2, 0.0, y2 - y1 ) == 0 )
        y2 = 0.0;

    return QRectF( x1, y1, x2 - x1 + 1, y2 - y1 + 1 );
}

/*!
   Transform a rectangle from paint to scale coordinates

   \param xMap X map
   \param yMap Y map
   \param pos Position in paint coordinates
   \return Position in scale coordinates
   \sa transform()
*/
QPointF QwtScaleMap::invTransform( const QwtScaleMap &xMap,
    const QwtScaleMap &yMap, const QPointF &pos )
{
    return QPointF( 
        xMap.invTransform( pos.x() ), 
        yMap.invTransform( pos.y() ) 
    );
}

/*!
   Transform a point from scale to paint coordinates

   \param xMap X map
   \param yMap Y map
   \param pos Position in scale coordinates
   \return Position in paint coordinates

   \sa invTransform()
*/
QPointF QwtScaleMap::transform( const QwtScaleMap &xMap,
    const QwtScaleMap &yMap, const QPointF &pos )
{
    return QPointF( 
        xMap.transform( pos.x() ), 
        yMap.transform( pos.y() )
    );
}

/*!
   Transform a rectangle from paint to scale coordinates

   \param xMap X map
   \param yMap Y map
   \param rect Rectangle in paint coordinates
   \return Rectangle in scale coordinates
   \sa transform()
*/
QRectF QwtScaleMap::invTransform( const QwtScaleMap &xMap,
    const QwtScaleMap &yMap, const QRectF &rect )
{
    const double x1 = xMap.invTransform( rect.left() );
    const double x2 = xMap.invTransform( rect.right() - 1 );
    const double y1 = yMap.invTransform( rect.top() );
    const double y2 = yMap.invTransform( rect.bottom() - 1 );

    const QRectF r( x1, y1, x2 - x1, y2 - y1 );
    return r.normalized();
}

#ifndef QT_NO_DEBUG_STREAM

QDebug operator<<( QDebug debug, const QwtScaleMap &map )
{
    debug.nospace() << "QwtScaleMap("
        << static_cast<int>( map.transformation()->type() )
        << ", s:" << map.s1() << "->" << map.s2()
        << ", p:" << map.p1() << "->" << map.p2()
        << ")";

    return debug.space();
}

#endif
