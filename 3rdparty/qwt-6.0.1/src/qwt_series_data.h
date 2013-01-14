/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_SERIES_DATA_H
#define QWT_SERIES_DATA_H 1

#include "qwt_global.h"
#include "qwt_interval.h"
#include "qwt_point_3d.h"
#include "qwt_point_polar.h"
#include <qvector.h>
#include <qrect.h>

//! \brief A sample of the types (x1-x2, y) or (x, y1-y2)
class QWT_EXPORT QwtIntervalSample
{
public:
    QwtIntervalSample();
    QwtIntervalSample( double, const QwtInterval & );
    QwtIntervalSample( double value, double min, double max );

    bool operator==( const QwtIntervalSample & ) const;
    bool operator!=( const QwtIntervalSample & ) const;

    //! Value
    double value;

    //! Interval
    QwtInterval interval;
};

/*!
  Constructor
  The value is set to 0.0, the interval is invalid
*/
inline QwtIntervalSample::QwtIntervalSample():
    value( 0.0 )
{
}

//! Constructor
inline QwtIntervalSample::QwtIntervalSample(
        double v, const QwtInterval &intv ):
    value( v ),
    interval( intv )
{
}

//! Constructor
inline QwtIntervalSample::QwtIntervalSample(
        double v, double min, double max ):
    value( v ),
    interval( min, max )
{
}

//! Compare operator
inline bool QwtIntervalSample::operator==( 
    const QwtIntervalSample &other ) const
{
    return value == other.value && interval == other.interval;
}

//! Compare operator
inline bool QwtIntervalSample::operator!=( 
    const QwtIntervalSample &other ) const
{
    return !( *this == other );
}

//! \brief A sample of the types (x1...xn, y) or (x, y1..yn)
class QWT_EXPORT QwtSetSample
{
public:
    QwtSetSample();
    bool operator==( const QwtSetSample &other ) const;
    bool operator!=( const QwtSetSample &other ) const;

    //! value
    double value;

    //! Vector of values associated to value
    QVector<double> set;
};

/*!
  Constructor
  The value is set to 0.0
*/
inline QwtSetSample::QwtSetSample():
    value( 0.0 )
{
}

//! Compare operator
inline bool QwtSetSample::operator==( const QwtSetSample &other ) const
{
    return value == other.value && set == other.set;
}

//! Compare operator
inline bool QwtSetSample::operator!=( const QwtSetSample &other ) const
{
    return !( *this == other );
}

/*!
   \brief Abstract interface for iterating over samples

   Qwt offers several implementations of the QwtSeriesData API,
   but in situations, where data of an application specific format
   needs to be displayed, without having to copy it, it is recommended
   to implement an individual data access.
*/
template <typename T>
class QwtSeriesData
{
public:
    QwtSeriesData();
    virtual ~QwtSeriesData();

    //! \return Number of samples
    virtual size_t size() const = 0;

    /*!
      Return a sample
      \param i Index
      \return Sample at position i
     */
    virtual T sample( size_t i ) const = 0;

    /*!
       Calculate the bounding rect of all samples

       The bounding rect is necessary for autoscaling and can be used
       for a couple of painting optimizations.

       qwtBoundingRect(...) offers slow implementations iterating
       over the samples. For large sets it is recommended to implement
       something faster f.e. by caching the bounding rect.
     */
    virtual QRectF boundingRect() const = 0;

    virtual void setRectOfInterest( const QRectF & );

protected:
    //! Can be used to cache a calculated bounding rectangle
    mutable QRectF d_boundingRect;

private:
    QwtSeriesData<T> &operator=( const QwtSeriesData<T> & );
};

//! Constructor
template <typename T>
QwtSeriesData<T>::QwtSeriesData():
    d_boundingRect( 0.0, 0.0, -1.0, -1.0 )
{
}

//! Destructor
template <typename T>
QwtSeriesData<T>::~QwtSeriesData()
{
}

/*!
   Set a the "rect of interest"

   QwtPlotSeriesItem defines the current area of the plot canvas
   as "rect of interest" ( QwtPlotSeriesItem::updateScaleDiv() ).
   It can be used to implement different levels of details.

   The default implementation does nothing.
*/
template <typename T>
void QwtSeriesData<T>::setRectOfInterest( const QRectF & )
{
}

/*!
  \brief Template class for data, that is organized as QVector

  QVector uses implicit data sharing and can be
  passed around as argument efficiently.
*/
template <typename T>
class QwtArraySeriesData: public QwtSeriesData<T>
{
public:
    QwtArraySeriesData();
    QwtArraySeriesData( const QVector<T> & );

    void setSamples( const QVector<T> & );
    const QVector<T> samples() const;

    virtual size_t size() const;
    virtual T sample( size_t ) const;

protected:
    //! Vector of samples
    QVector<T> d_samples;
};

//! Constructor
template <typename T>
QwtArraySeriesData<T>::QwtArraySeriesData()
{
}

/*!
   Constructor
   \param samples Array of samples
*/
template <typename T>
QwtArraySeriesData<T>::QwtArraySeriesData( const QVector<T> &samples ):
    d_samples( samples )
{
}

/*!
  Assign an array of samples
  \param samples Array of samples
*/
template <typename T>
void QwtArraySeriesData<T>::setSamples( const QVector<T> &samples )
{
    QwtSeriesData<T>::d_boundingRect = QRectF( 0.0, 0.0, -1.0, -1.0 );
    d_samples = samples;
}

//! \return Array of samples
template <typename T>
const QVector<T> QwtArraySeriesData<T>::samples() const
{
    return d_samples;
}

//! \return Number of samples
template <typename T>
size_t QwtArraySeriesData<T>::size() const
{
    return d_samples.size();
}

/*!
  Return a sample
  \param i Index
  \return Sample at position i
*/
template <typename T>
T QwtArraySeriesData<T>::sample( size_t i ) const
{
    return d_samples[i];
}

//! Interface for iterating over an array of points
class QWT_EXPORT QwtPointSeriesData: public QwtArraySeriesData<QPointF>
{
public:
    QwtPointSeriesData(
        const QVector<QPointF> & = QVector<QPointF>() );

    virtual QRectF boundingRect() const;
};

//! Interface for iterating over an array of 3D points
class QWT_EXPORT QwtPoint3DSeriesData: public QwtArraySeriesData<QwtPoint3D>
{
public:
    QwtPoint3DSeriesData(
        const QVector<QwtPoint3D> & = QVector<QwtPoint3D>() );
    virtual QRectF boundingRect() const;
};

//! Interface for iterating over an array of intervals
class QWT_EXPORT QwtIntervalSeriesData: public QwtArraySeriesData<QwtIntervalSample>
{
public:
    QwtIntervalSeriesData(
        const QVector<QwtIntervalSample> & = QVector<QwtIntervalSample>() );

    virtual QRectF boundingRect() const;
};

//! Interface for iterating over an array of samples
class QWT_EXPORT QwtSetSeriesData: public QwtArraySeriesData<QwtSetSample>
{
public:
    QwtSetSeriesData(
        const QVector<QwtSetSample> & = QVector<QwtSetSample>() );

    virtual QRectF boundingRect() const;
};

/*!
  \brief Interface for iterating over two QVector<double> objects.
*/
class QWT_EXPORT QwtPointArrayData: public QwtSeriesData<QPointF>
{
public:
    QwtPointArrayData( const QVector<double> &x, const QVector<double> &y );
    QwtPointArrayData( const double *x, const double *y, size_t size );

    virtual QRectF boundingRect() const;

    virtual size_t size() const;
    virtual QPointF sample( size_t i ) const;

    const QVector<double> &xData() const;
    const QVector<double> &yData() const;

private:
    QVector<double> d_x;
    QVector<double> d_y;
};

/*!
  \brief Data class containing two pointers to memory blocks of doubles.
 */
class QWT_EXPORT QwtCPointerData: public QwtSeriesData<QPointF>
{
public:
    QwtCPointerData( const double *x, const double *y, size_t size );

    virtual QRectF boundingRect() const;
    virtual size_t size() const;
    virtual QPointF sample( size_t i ) const;

    const double *xData() const;
    const double *yData() const;

private:
    const double *d_x;
    const double *d_y;
    size_t d_size;
};

/*!
  \brief Synthetic point data

  QwtSyntheticPointData provides a fixed number of points for an interval.
  The points are calculated in equidistant steps in x-direction.

  If the interval is invalid, the points are calculated for
  the "rect of interest", what normally is the displayed area on the
  plot canvas. In this mode you get different levels of detail, when
  zooming in/out.

  \par Example

  The following example shows how to implement a sinus curve.

  \verbatim
#include <cmath>
#include <qwt_series_data.h>
#include <qwt_plot_curve.h>
#include <qwt_plot.h>
#include <qapplication.h>

class SinusData: public QwtSyntheticPointData
{
public:
    SinusData():
        QwtSyntheticPointData(100)
    {
    }
    virtual double y(double x) const
    {
        return qSin(x);
    }
};

int main(int argc, char **argv)
{
    QApplication a(argc, argv);

    QwtPlot plot;
    plot.setAxisScale(QwtPlot::xBottom, 0.0, 10.0);
    plot.setAxisScale(QwtPlot::yLeft, -1.0, 1.0);

    QwtPlotCurve *curve = new QwtPlotCurve("y = sin(x)");
    curve->setData(SinusData());
    curve->attach(&plot);

    plot.show();
    return a.exec();
}
   \endverbatim
*/
class QWT_EXPORT QwtSyntheticPointData: public QwtSeriesData<QPointF>
{
public:
    QwtSyntheticPointData( size_t size,
        const QwtInterval & = QwtInterval() );

    void setSize( size_t size );
    size_t size() const;

    void setInterval( const QwtInterval& );
    QwtInterval interval() const;

    virtual QRectF boundingRect() const;
    virtual QPointF sample( size_t i ) const;

    /*!
       Calculate a y value for a x value

       \param x x value
       \return Corresponding y value
     */
    virtual double y( double x ) const = 0;
    virtual double x( uint index ) const;

    virtual void setRectOfInterest( const QRectF & );
    QRectF rectOfInterest() const;

private:
    size_t d_size;
    QwtInterval d_interval;
    QRectF d_rectOfInterest;
    QwtInterval d_intervalOfInterest;
};

QWT_EXPORT QRectF qwtBoundingRect(
    const QwtSeriesData<QPointF> &, int from = 0, int to = -1 );
QWT_EXPORT QRectF qwtBoundingRect(
    const QwtSeriesData<QwtPoint3D> &, int from = 0, int to = -1 );
QWT_EXPORT QRectF qwtBoundingRect(
    const QwtSeriesData<QwtPointPolar> &, int from = 0, int to = -1 );
QWT_EXPORT QRectF qwtBoundingRect(
    const QwtSeriesData<QwtIntervalSample> &, int from = 0, int to = -1 );
QWT_EXPORT QRectF qwtBoundingRect(
    const QwtSeriesData<QwtSetSample> &, int from = 0, int to = -1 );

#endif 
