/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_PLOT_SERIES_ITEM_H
#define QWT_PLOT_SERIES_ITEM_H

#include "qwt_global.h"
#include "qwt_plot_item.h"
#include "qwt_scale_div.h"
#include "qwt_series_data.h"

/*!
  \brief Base class for plot items representing a series of samples
*/
class QWT_EXPORT QwtPlotAbstractSeriesItem: public QwtPlotItem
{
public:
    explicit QwtPlotAbstractSeriesItem( const QString &title = QString::null );
    explicit QwtPlotAbstractSeriesItem( const QwtText &title );

    virtual ~QwtPlotAbstractSeriesItem();

    void setOrientation( Qt::Orientation );
    Qt::Orientation orientation() const;

    virtual void draw( QPainter *p,
        const QwtScaleMap &xMap, const QwtScaleMap &yMap,
        const QRectF & ) const;

    /*!
      Draw a subset of the samples

      \param painter Painter
      \param xMap Maps x-values into pixel coordinates.
      \param yMap Maps y-values into pixel coordinates.
      \param canvasRect Contents rect of the canvas
      \param from Index of the first point to be painted
      \param to Index of the last point to be painted. If to < 0 the
             curve will be painted to its last point.
    */
    virtual void drawSeries( QPainter *painter,
        const QwtScaleMap &xMap, const QwtScaleMap &yMap,
        const QRectF &canvasRect, int from, int to ) const = 0;

private:
    class PrivateData;
    PrivateData *d_data;
};

/*!
  \brief Class template for plot items representing a series of samples
*/
template <typename T>
class QwtPlotSeriesItem: public QwtPlotAbstractSeriesItem
{
public:
    explicit QwtPlotSeriesItem<T>( const QString &title = QString::null );
    explicit QwtPlotSeriesItem<T>( const QwtText &title );

    virtual ~QwtPlotSeriesItem<T>();

    void setData( QwtSeriesData<T> * );

    QwtSeriesData<T> *data();
    const QwtSeriesData<T> *data() const;

    size_t dataSize() const;
    T sample( int index ) const;

    virtual QRectF boundingRect() const;
    virtual void updateScaleDiv( const QwtScaleDiv &,
                                 const QwtScaleDiv & );

protected:
    //! Series
    QwtSeriesData<T> *d_series;
};

/*!
  Constructor
  \param title Title of the series item
*/
template <typename T>
QwtPlotSeriesItem<T>::QwtPlotSeriesItem( const QString &title ):
    QwtPlotAbstractSeriesItem( QwtText( title ) ),
    d_series( NULL )
{
}

/*!
  Constructor
  \param title Title of the series item
*/
template <typename T>
QwtPlotSeriesItem<T>::QwtPlotSeriesItem( const QwtText &title ):
    QwtPlotAbstractSeriesItem( title ),
    d_series( NULL )
{
}

//! Destructor
template <typename T>
QwtPlotSeriesItem<T>::~QwtPlotSeriesItem()
{
    delete d_series;
}

//! \return the the curve data
template <typename T>
inline QwtSeriesData<T> *QwtPlotSeriesItem<T>::data()
{
    return d_series;
}

//! \return the the curve data
template <typename T>
inline const QwtSeriesData<T> *QwtPlotSeriesItem<T>::data() const
{
    return d_series;
}

/*!
    \param index Index
    \return Sample at position index
*/
template <typename T>
inline T QwtPlotSeriesItem<T>::sample( int index ) const
{
    return d_series ? d_series->sample( index ) : T();
}

/*!
  Assign a series of samples

  \param data Data
  \warning The item takes ownership of the data object, deleting
           it when its not used anymore.
*/
template <typename T>
void QwtPlotSeriesItem<T>::setData( QwtSeriesData<T> *data )
{
    if ( d_series != data )
    {
        delete d_series;
        d_series = data;
        itemChanged();
    }
}

/*!
  Return the size of the data arrays
  \sa setData()
*/
template <typename T>
size_t QwtPlotSeriesItem<T>::dataSize() const
{
    if ( d_series == NULL )
        return 0;

    return d_series->size();
}

/*!
  \return Bounding rectangle of the data.
  If there is no bounding rect, like for empty data the rectangle is invalid.

  \sa QwtSeriesData<T>::boundingRect(), QRectF::isValid()
*/
template <typename T>
QRectF QwtPlotSeriesItem<T>::boundingRect() const
{
    if ( d_series == NULL )
        return QRectF( 1.0, 1.0, -2.0, -2.0 ); // invalid

    return d_series->boundingRect();
}

/*!
   Update the rect of interest according to the current scale ranges

   \param xScaleDiv Scale division of the x-axis
   \param yScaleDiv Scale division of the y-axis

   \sa QwtSeriesData<T>::setRectOfInterest()
*/
template <typename T>
void QwtPlotSeriesItem<T>::updateScaleDiv( 
    const QwtScaleDiv &xScaleDiv, const QwtScaleDiv &yScaleDiv )
{
    const QRectF rect = QRectF(
        xScaleDiv.lowerBound(), yScaleDiv.lowerBound(),
        xScaleDiv.range(), yScaleDiv.range() );

    d_series->setRectOfInterest( rect );
}

#endif
