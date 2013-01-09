/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_plot_picker.h"
#include "qwt_plot.h"
#include "qwt_scale_div.h"
#include "qwt_painter.h"
#include "qwt_scale_map.h"
#include "qwt_picker_machine.h"

/*!
  \brief Create a plot picker

  The picker is set to those x- and y-axis of the plot
  that are enabled. If both or no x-axis are enabled, the picker
  is set to QwtPlot::xBottom. If both or no y-axis are
  enabled, it is set to QwtPlot::yLeft.

  \param canvas Plot canvas to observe, also the parent object

  \sa QwtPlot::autoReplot(), QwtPlot::replot(), scaleRect()
*/

QwtPlotPicker::QwtPlotPicker( QwtPlotCanvas *canvas ):
    QwtPicker( canvas ),
    d_xAxis( -1 ),
    d_yAxis( -1 )
{
    if ( !canvas )
        return;

    // attach axes

    int xAxis = QwtPlot::xBottom;

    const QwtPlot *plot = QwtPlotPicker::plot();
    if ( !plot->axisEnabled( QwtPlot::xBottom ) &&
        plot->axisEnabled( QwtPlot::xTop ) )
    {
        xAxis = QwtPlot::xTop;
    }

    int yAxis = QwtPlot::yLeft;
    if ( !plot->axisEnabled( QwtPlot::yLeft ) &&
        plot->axisEnabled( QwtPlot::yRight ) )
    {
        yAxis = QwtPlot::yRight;
    }

    setAxis( xAxis, yAxis );
}

/*!
  Create a plot picker

  \param xAxis Set the x axis of the picker
  \param yAxis Set the y axis of the picker
  \param canvas Plot canvas to observe, also the parent object

  \sa QwtPlot::autoReplot(), QwtPlot::replot(), scaleRect()
*/
QwtPlotPicker::QwtPlotPicker( int xAxis, int yAxis, QwtPlotCanvas *canvas ):
    QwtPicker( canvas ),
    d_xAxis( xAxis ),
    d_yAxis( yAxis )
{
}

/*!
  Create a plot picker

  \param xAxis X axis of the picker
  \param yAxis Y axis of the picker
  \param rubberBand Rubberband style
  \param trackerMode Tracker mode
  \param canvas Plot canvas to observe, also the parent object

  \sa QwtPicker, QwtPicker::setSelectionFlags(), QwtPicker::setRubberBand(),
      QwtPicker::setTrackerMode

  \sa QwtPlot::autoReplot(), QwtPlot::replot(), scaleRect()
*/
QwtPlotPicker::QwtPlotPicker( int xAxis, int yAxis,
        RubberBand rubberBand, DisplayMode trackerMode,
        QwtPlotCanvas *canvas ):
    QwtPicker( rubberBand, trackerMode, canvas ),
    d_xAxis( xAxis ),
    d_yAxis( yAxis )
{
}

//! Destructor
QwtPlotPicker::~QwtPlotPicker()
{
}

//! Return observed plot canvas
QwtPlotCanvas *QwtPlotPicker::canvas()
{
    return qobject_cast<QwtPlotCanvas *>( parentWidget() );
}

//! Return Observed plot canvas
const QwtPlotCanvas *QwtPlotPicker::canvas() const
{
    return qobject_cast<const QwtPlotCanvas *>( parentWidget() );
}

//! Return plot widget, containing the observed plot canvas
QwtPlot *QwtPlotPicker::plot()
{
    QwtPlotCanvas *w = canvas();
    if ( w )
        return w->plot();

    return NULL;
}

//! Return plot widget, containing the observed plot canvas
const QwtPlot *QwtPlotPicker::plot() const
{
    const QwtPlotCanvas *w = canvas();
    if ( w )
        return w->plot();

    return NULL;
}

/*!
  Return normalized bounding rect of the axes

  \sa QwtPlot::autoReplot(), QwtPlot::replot().
*/
QRectF QwtPlotPicker::scaleRect() const
{
    QRectF rect;

    if ( plot() )
    {
        const QwtScaleDiv *xs = plot()->axisScaleDiv( xAxis() );
        const QwtScaleDiv *ys = plot()->axisScaleDiv( yAxis() );

        if ( xs && ys )
        {
            rect = QRectF( xs->lowerBound(), ys->lowerBound(),
                xs->range(), ys->range() );
            rect = rect.normalized();
        }
    }

    return rect;
}

/*!
  Set the x and y axes of the picker

  \param xAxis X axis
  \param yAxis Y axis
*/
void QwtPlotPicker::setAxis( int xAxis, int yAxis )
{
    const QwtPlot *plt = plot();
    if ( !plt )
        return;

    if ( xAxis != d_xAxis || yAxis != d_yAxis )
    {
        d_xAxis = xAxis;
        d_yAxis = yAxis;
    }
}

//! Return x axis
int QwtPlotPicker::xAxis() const
{
    return d_xAxis;
}

//! Return y axis
int QwtPlotPicker::yAxis() const
{
    return d_yAxis;
}

/*!
  Translate a pixel position into a position string

  \param pos Position in pixel coordinates
  \return Position string
*/
QwtText QwtPlotPicker::trackerText( const QPoint &pos ) const
{
    return trackerTextF( invTransform( pos ) );
}

/*!
  \brief Translate a position into a position string

  In case of HLineRubberBand the label is the value of the
  y position, in case of VLineRubberBand the value of the x position.
  Otherwise the label contains x and y position separated by a ',' .

  The format for the double to string conversion is "%.4f".

  \param pos Position
  \return Position string
*/
QwtText QwtPlotPicker::trackerTextF( const QPointF &pos ) const
{
    QString text;

    switch ( rubberBand() )
    {
        case HLineRubberBand:
            text.sprintf( "%.4f", pos.y() );
            break;
        case VLineRubberBand:
            text.sprintf( "%.4f", pos.x() );
            break;
        default:
            text.sprintf( "%.4f, %.4f", pos.x(), pos.y() );
    }
    return QwtText( text );
}

/*!
  Append a point to the selection and update rubberband and tracker.

  \param pos Additional point
  \sa isActive, begin(), end(), move(), appended()

  \note The appended(const QPoint &), appended(const QDoublePoint &)
        signals are emitted.
*/
void QwtPlotPicker::append( const QPoint &pos )
{
    QwtPicker::append( pos );
    Q_EMIT appended( invTransform( pos ) );
}

/*!
  Move the last point of the selection

  \param pos New position
  \sa isActive, begin(), end(), append()

  \note The moved(const QPoint &), moved(const QDoublePoint &)
        signals are emitted.
*/
void QwtPlotPicker::move( const QPoint &pos )
{
    QwtPicker::move( pos );
    Q_EMIT moved( invTransform( pos ) );
}

/*!
  Close a selection setting the state to inactive.

  \param ok If true, complete the selection and emit selected signals
            otherwise discard the selection.
  \return true if the selection is accepted, false otherwise
*/

bool QwtPlotPicker::end( bool ok )
{
    ok = QwtPicker::end( ok );
    if ( !ok )
        return false;

    QwtPlot *plot = QwtPlotPicker::plot();
    if ( !plot )
        return false;

    const QPolygon pa = selection();
    if ( pa.count() == 0 )
        return false;

    QwtPickerMachine::SelectionType selectionType =
        QwtPickerMachine::NoSelection;

    if ( stateMachine() )
        selectionType = stateMachine()->selectionType();

    switch ( selectionType )
    {
        case QwtPickerMachine::PointSelection:
        {
            const QPointF pos = invTransform( pa[0] );
            Q_EMIT selected( pos );
            break;
        }
        case QwtPickerMachine::RectSelection:
        {
            if ( pa.count() >= 2 )
            {
                const QPoint p1 = pa[0];
                const QPoint p2 = pa[int( pa.count() - 1 )];

                const QRect rect = QRect( p1, p2 ).normalized();
                Q_EMIT selected( invTransform( rect ) );
            }
            break;
        }
        case QwtPickerMachine::PolygonSelection:
        {
            QVector<QPointF> dpa( pa.count() );
            for ( int i = 0; i < int( pa.count() ); i++ )
                dpa[i] = invTransform( pa[i] );

            Q_EMIT selected( dpa );
        }
        default:
            break;
    }

    return true;
}

/*!
    Translate a rectangle from pixel into plot coordinates

    \return Rectangle in plot coordinates
    \sa transform()
*/
QRectF QwtPlotPicker::invTransform( const QRect &rect ) const
{
    const QwtScaleMap xMap = plot()->canvasMap( d_xAxis );
    const QwtScaleMap yMap = plot()->canvasMap( d_yAxis );

    return QwtScaleMap::invTransform( xMap, yMap, rect );
}

/*!
    Translate a rectangle from plot into pixel coordinates
    \return Rectangle in pixel coordinates
    \sa invTransform()
*/
QRect QwtPlotPicker::transform( const QRectF &rect ) const
{
    const QwtScaleMap xMap = plot()->canvasMap( d_xAxis );
    const QwtScaleMap yMap = plot()->canvasMap( d_yAxis );

    return QwtScaleMap::transform( xMap, yMap, rect ).toRect();
}

/*!
    Translate a point from pixel into plot coordinates
    \return Point in plot coordinates
    \sa transform()
*/
QPointF QwtPlotPicker::invTransform( const QPoint &pos ) const
{
    QwtScaleMap xMap = plot()->canvasMap( d_xAxis );
    QwtScaleMap yMap = plot()->canvasMap( d_yAxis );

    return QPointF(
        xMap.invTransform( pos.x() ),
        yMap.invTransform( pos.y() )
    );
}

/*!
    Translate a point from plot into pixel coordinates
    \return Point in pixel coordinates
    \sa invTransform()
*/
QPoint QwtPlotPicker::transform( const QPointF &pos ) const
{
    QwtScaleMap xMap = plot()->canvasMap( d_xAxis );
    QwtScaleMap yMap = plot()->canvasMap( d_yAxis );

    const QPointF p( xMap.transform( pos.x() ),
        yMap.transform( pos.y() ) );

    return p.toPoint();
}
