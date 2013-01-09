/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_series_data.h"
#include "qwt_math.h"

static inline QRectF qwtBoundingRect( const QPointF &sample )
{
    return QRectF( sample.x(), sample.y(), 0.0, 0.0 );
}

static inline QRectF qwtBoundingRect( const QwtPoint3D &sample )
{
    return QRectF( sample.x(), sample.y(), 0.0, 0.0 );
}

static inline QRectF qwtBoundingRect( const QwtPointPolar &sample )
{
    return QRectF( sample.azimuth(), sample.radius(), 0.0, 0.0 );
}

static inline QRectF qwtBoundingRect( const QwtIntervalSample &sample )
{
    return QRectF( sample.interval.minValue(), sample.value,
        sample.interval.maxValue() - sample.interval.minValue(), 0.0 );
}

static inline QRectF qwtBoundingRect( const QwtSetSample &sample )
{
    double minX = sample.set[0];
    double maxX = sample.set[0];

    for ( int i = 1; i < ( int )sample.set.size(); i++ )
    {
        if ( sample.set[i] < minX )
            minX = sample.set[i];
        if ( sample.set[i] > maxX )
            maxX = sample.set[i];
    }

    double minY = sample.value;
    double maxY = sample.value;

    return QRectF( minX, minY, maxX - minX, maxY - minY );
}

/*!
  \brief Calculate the bounding rect of a series subset

  Slow implementation, that iterates over the series.

  \param series Series
  \param from Index of the first sample, <= 0 means from the beginning
  \param to Index of the last sample, < 0 means to the end

  \return Bounding rectangle
*/

template <class T>
QRectF qwtBoundingRectT( 
    const QwtSeriesData<T>& series, int from, int to )
{
    QRectF boundingRect( 1.0, 1.0, -2.0, -2.0 ); // invalid;

    if ( from < 0 )
        from = 0;

    if ( to < 0 )
        to = series.size() - 1;

    if ( to < from )
        return boundingRect;

    int i;
    for ( i = from; i <= to; i++ )
    {
        const QRectF rect = qwtBoundingRect( series.sample( i ) );
        if ( rect.width() >= 0.0 && rect.height() >= 0.0 )
        {
            boundingRect = rect;
            i++;
            break;
        }
    }

    for ( ; i <= to; i++ )
    {
        const QRectF rect = qwtBoundingRect( series.sample( i ) );
        if ( rect.width() >= 0.0 && rect.height() >= 0.0 )
        {
            boundingRect.setLeft( qMin( boundingRect.left(), rect.left() ) );
            boundingRect.setRight( qMax( boundingRect.right(), rect.right() ) );
            boundingRect.setTop( qMin( boundingRect.top(), rect.top() ) );
            boundingRect.setBottom( qMax( boundingRect.bottom(), rect.bottom() ) );
        }
    }

    return boundingRect;
}

/*!
  \brief Calculate the bounding rect of a series subset

  Slow implementation, that iterates over the series.

  \param series Series
  \param from Index of the first sample, <= 0 means from the beginning
  \param to Index of the last sample, < 0 means to the end

  \return Bounding rectangle
*/
QRectF qwtBoundingRect( 
    const QwtSeriesData<QPointF> &series, int from, int to )
{
    return qwtBoundingRectT<QPointF>( series, from, to );
}

/*!
  \brief Calculate the bounding rect of a series subset

  Slow implementation, that iterates over the series.

  \param series Series
  \param from Index of the first sample, <= 0 means from the beginning
  \param to Index of the last sample, < 0 means to the end

  \return Bounding rectangle
*/
QRectF qwtBoundingRect( 
    const QwtSeriesData<QwtPoint3D> &series, int from, int to )
{
    return qwtBoundingRectT<QwtPoint3D>( series, from, to );
}

/*!
  \brief Calculate the bounding rect of a series subset

  The horizontal coordinates represent the azimuth, the
  vertical coordinates the radius.

  Slow implementation, that iterates over the series.

  \param series Series
  \param from Index of the first sample, <= 0 means from the beginning
  \param to Index of the last sample, < 0 means to the end

  \return Bounding rectangle
*/
QRectF qwtBoundingRect( 
    const QwtSeriesData<QwtPointPolar> &series, int from, int to )
{
    return qwtBoundingRectT<QwtPointPolar>( series, from, to );
}

/*!
  \brief Calculate the bounding rect of a series subset

  Slow implementation, that iterates over the series.

  \param series Series
  \param from Index of the first sample, <= 0 means from the beginning
  \param to Index of the last sample, < 0 means to the end

  \return Bounding rectangle
*/
QRectF qwtBoundingRect( 
    const QwtSeriesData<QwtIntervalSample>& series, int from, int to )
{
    return qwtBoundingRectT<QwtIntervalSample>( series, from, to );
}

/*!
  \brief Calculate the bounding rect of a series subset

  Slow implementation, that iterates over the series.

  \param series Series
  \param from Index of the first sample, <= 0 means from the beginning
  \param to Index of the last sample, < 0 means to the end

  \return Bounding rectangle
*/
QRectF qwtBoundingRect( 
    const QwtSeriesData<QwtSetSample>& series, int from, int to )
{
    return qwtBoundingRectT<QwtSetSample>( series, from, to );
}

/*!
   Constructor
   \param samples Samples
*/
QwtPointSeriesData::QwtPointSeriesData(
        const QVector<QPointF> &samples ):
    QwtArraySeriesData<QPointF>( samples )
{
}

/*!
  \brief Calculate the bounding rect

  The bounding rectangle is calculated once by iterating over all
  points and is stored for all following requests.

  \return Bounding rectangle
*/
QRectF QwtPointSeriesData::boundingRect() const
{
    if ( d_boundingRect.width() < 0.0 )
        d_boundingRect = qwtBoundingRect( *this );

    return d_boundingRect;
}

/*!
   Constructor
   \param samples Samples
*/
QwtPoint3DSeriesData::QwtPoint3DSeriesData(
        const QVector<QwtPoint3D> &samples ):
    QwtArraySeriesData<QwtPoint3D>( samples )
{
}

/*!
  \brief Calculate the bounding rect

  The bounding rectangle is calculated once by iterating over all
  points and is stored for all following requests.

  \return Bounding rectangle
*/
QRectF QwtPoint3DSeriesData::boundingRect() const
{
    if ( d_boundingRect.width() < 0.0 )
        d_boundingRect = qwtBoundingRect( *this );

    return d_boundingRect;
}

/*!
   Constructor
   \param samples Samples
*/
QwtIntervalSeriesData::QwtIntervalSeriesData(
        const QVector<QwtIntervalSample> &samples ):
    QwtArraySeriesData<QwtIntervalSample>( samples )
{
}

/*!
  \brief Calculate the bounding rect

  The bounding rectangle is calculated once by iterating over all
  points and is stored for all following requests.

  \return Bounding rectangle
*/
QRectF QwtIntervalSeriesData::boundingRect() const
{
    if ( d_boundingRect.width() < 0.0 )
        d_boundingRect = qwtBoundingRect( *this );

    return d_boundingRect;
}

/*!
   Constructor
   \param samples Samples
*/
QwtSetSeriesData::QwtSetSeriesData(
        const QVector<QwtSetSample> &samples ):
    QwtArraySeriesData<QwtSetSample>( samples )
{
}

/*!
  \brief Calculate the bounding rect

  The bounding rectangle is calculated once by iterating over all
  points and is stored for all following requests.

  \return Bounding rectangle
*/
QRectF QwtSetSeriesData::boundingRect() const
{
    if ( d_boundingRect.width() < 0.0 )
        d_boundingRect = qwtBoundingRect( *this );

    return d_boundingRect;
}

/*!
  Constructor

  \param x Array of x values
  \param y Array of y values

  \sa QwtPlotCurve::setData(), QwtPlotCurve::setSamples()
*/
QwtPointArrayData::QwtPointArrayData(
        const QVector<double> &x, const QVector<double> &y ):
    d_x( x ),
    d_y( y )
{
}

/*!
  Constructor

  \param x Array of x values
  \param y Array of y values
  \param size Size of the x and y arrays
  \sa QwtPlotCurve::setData(), QwtPlotCurve::setSamples()
*/
QwtPointArrayData::QwtPointArrayData( const double *x,
        const double *y, size_t size )
{
    d_x.resize( size );
    qMemCopy( d_x.data(), x, size * sizeof( double ) );

    d_y.resize( size );
    qMemCopy( d_y.data(), y, size * sizeof( double ) );
}

/*!
  \brief Calculate the bounding rect

  The bounding rectangle is calculated once by iterating over all
  points and is stored for all following requests.

  \return Bounding rectangle
*/
QRectF QwtPointArrayData::boundingRect() const
{
    if ( d_boundingRect.width() < 0 )
        d_boundingRect = qwtBoundingRect( *this );

    return d_boundingRect;
}

//! \return Size of the data set
size_t QwtPointArrayData::size() const
{
    return qMin( d_x.size(), d_y.size() );
}

/*!
  Return the sample at position i

  \param i Index
  \return Sample at position i
*/
QPointF QwtPointArrayData::sample( size_t i ) const
{
    return QPointF( d_x[int( i )], d_y[int( i )] );
}

//! \return Array of the x-values
const QVector<double> &QwtPointArrayData::xData() const
{
    return d_x;
}

//! \return Array of the y-values
const QVector<double> &QwtPointArrayData::yData() const
{
    return d_y;
}

/*!
  Constructor

  \param x Array of x values
  \param y Array of y values
  \param size Size of the x and y arrays

  \warning The programmer must assure that the memory blocks referenced
           by the pointers remain valid during the lifetime of the
           QwtPlotCPointer object.

  \sa QwtPlotCurve::setData(), QwtPlotCurve::setRawSamples()
*/
QwtCPointerData::QwtCPointerData(
        const double *x, const double *y, size_t size ):
    d_x( x ),
    d_y( y ),
    d_size( size )
{
}

/*!
  \brief Calculate the bounding rect

  The bounding rectangle is calculated once by iterating over all
  points and is stored for all following requests.

  \return Bounding rectangle
*/
QRectF QwtCPointerData::boundingRect() const
{
    if ( d_boundingRect.width() < 0 )
        d_boundingRect = qwtBoundingRect( *this );

    return d_boundingRect;
}

//! \return Size of the data set
size_t QwtCPointerData::size() const
{
    return d_size;
}

/*!
  Return the sample at position i

  \param i Index
  \return Sample at position i
*/
QPointF QwtCPointerData::sample( size_t i ) const
{
    return QPointF( d_x[int( i )], d_y[int( i )] );
}

//! \return Array of the x-values
const double *QwtCPointerData::xData() const
{
    return d_x;
}

//! \return Array of the y-values
const double *QwtCPointerData::yData() const
{
    return d_y;
}

/*!
   Constructor

   \param size Number of points
   \param interval Bounding interval for the points

   \sa setInterval(), setSize()
*/
QwtSyntheticPointData::QwtSyntheticPointData(
        size_t size, const QwtInterval &interval ):
    d_size( size ),
    d_interval( interval )
{
}

/*!
  Change the number of points

  \param size Number of points
  \sa size(), setInterval()
*/
void QwtSyntheticPointData::setSize( size_t size )
{
    d_size = size;
}

/*!
  \return Number of points
  \sa setSize(), interval()
*/
size_t QwtSyntheticPointData::size() const
{
    return d_size;
}

/*!
   Set the bounding interval

   \param interval Interval
   \sa interval(), setSize()
*/
void QwtSyntheticPointData::setInterval( const QwtInterval &interval )
{
    d_interval = interval.normalized();
}

/*!
   \return Bounding interval
   \sa setInterval(), size()
*/
QwtInterval QwtSyntheticPointData::interval() const
{
    return d_interval;
}

/*!
   Set a the "rect of interest"

   QwtPlotSeriesItem defines the current area of the plot canvas
   as "rect of interest" ( QwtPlotSeriesItem::updateScaleDiv() ).

   If interval().isValid() == false the x values are calculated
   in the interval rect.left() -> rect.right().

   \sa rectOfInterest()
*/
void QwtSyntheticPointData::setRectOfInterest( const QRectF &rect )
{
    d_rectOfInterest = rect;
    d_intervalOfInterest = QwtInterval(
        rect.left(), rect.right() ).normalized();
}

/*!
   \return "rect of interest"
   \sa setRectOfInterest()
*/
QRectF QwtSyntheticPointData::rectOfInterest() const
{
    return d_rectOfInterest;
}

/*!
  \brief Calculate the bounding rect

  This implementation iterates over all points, what could often
  be implemented much faster using the characteristics of the series.
  When there are many points it is recommended to overload and
  reimplement this method using the characteristics of the series
  ( if possible ).

  \return Bounding rectangle
*/
QRectF QwtSyntheticPointData::boundingRect() const
{
    if ( d_size == 0 || 
        !( d_interval.isValid() || d_intervalOfInterest.isValid() ) )
    {
        return QRectF(1.0, 1.0, -2.0, -2.0); // something invalid
    }

    return qwtBoundingRect( *this );
}

/*!
   Calculate the point from an index

   \param index Index
   \return QPointF(x(index), y(x(index)));

   \warning For invalid indices ( index < 0 || index >= size() )
            (0, 0) is returned.
*/
QPointF QwtSyntheticPointData::sample( size_t index ) const
{
    if ( index >= d_size )
        return QPointF( 0, 0 );

    const double xValue = x( index );
    const double yValue = y( xValue );

    return QPointF( xValue, yValue );
}

/*!
   Calculate a x-value from an index

   x values are calculated by deviding an interval into
   equidistant steps. If !interval().isValid() the
   interval is calculated from the "rect of interest".

   \sa interval(), rectOfInterest(), y()
*/
double QwtSyntheticPointData::x( uint index ) const
{
    const QwtInterval &interval = d_interval.isValid() ?
        d_interval : d_intervalOfInterest;

    if ( !interval.isValid() || d_size == 0 || index >= d_size )
        return 0.0;

    const double dx = interval.width() / d_size;
    return interval.minValue() + index * dx;
}
