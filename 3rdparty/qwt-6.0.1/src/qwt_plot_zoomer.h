/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_PLOT_ZOOMER_H
#define QWT_PLOT_ZOOMER_H

#include "qwt_global.h"
#include "qwt_plot_picker.h"
#include <qstack.h>

/*!
  \brief QwtPlotZoomer provides stacked zooming for a plot widget

  QwtPlotZoomer offers rubberband selections on the plot canvas,
  translating the selected rectangles into plot coordinates and
  adjusting the axes to them. Zooming can repeated as often as
  possible, limited only by maxStackDepth() or minZoomSize().
  Each rectangle is pushed on a stack.

  Zoom rectangles can be selected depending on selectionFlags() using the
  mouse or keyboard (QwtEventPattern, QwtPickerMachine).
  QwtEventPattern::MouseSelect3,QwtEventPattern::KeyUndo,
  or QwtEventPattern::MouseSelect6,QwtEventPattern::KeyRedo
  walk up and down the zoom stack.
  QwtEventPattern::MouseSelect2 or QwtEventPattern::KeyHome unzoom to
  the initial size.

  QwtPlotZoomer is tailored for plots with one x and y axis, but it is
  allowed to attach a second QwtPlotZoomer for the other axes.

  \note The realtime example includes an derived zoomer class that adds
        scrollbars to the plot canvas.
*/

class QWT_EXPORT QwtPlotZoomer: public QwtPlotPicker
{
    Q_OBJECT
public:
    explicit QwtPlotZoomer( QwtPlotCanvas *, bool doReplot = true );
    explicit QwtPlotZoomer( int xAxis, int yAxis,
                            QwtPlotCanvas *, bool doReplot = true );

    virtual ~QwtPlotZoomer();

    virtual void setZoomBase( bool doReplot = true );
    virtual void setZoomBase( const QRectF & );

    QRectF zoomBase() const;
    QRectF zoomRect() const;

    virtual void setAxis( int xAxis, int yAxis );

    void setMaxStackDepth( int );
    int maxStackDepth() const;

    const QStack<QRectF> &zoomStack() const;
    void setZoomStack( const QStack<QRectF> &,
        int zoomRectIndex = -1 );

    uint zoomRectIndex() const;

public Q_SLOTS:
    void moveBy( double x, double y );
    virtual void moveTo( const QPointF & );

    virtual void zoom( const QRectF & );
    virtual void zoom( int up );

Q_SIGNALS:
    /*!
      A signal emitting the zoomRect(), when the plot has been
      zoomed in or out.

      \param rect Current zoom rectangle.
    */

    void zoomed( const QRectF &rect );

protected:
    virtual void rescale();

    virtual QSizeF minZoomSize() const;

    virtual void widgetMouseReleaseEvent( QMouseEvent * );
    virtual void widgetKeyPressEvent( QKeyEvent * );

    virtual void begin();
    virtual bool end( bool ok = true );
    virtual bool accept( QPolygon & ) const;

private:
    void init( bool doReplot );

    class PrivateData;
    PrivateData *d_data;
};

#endif
