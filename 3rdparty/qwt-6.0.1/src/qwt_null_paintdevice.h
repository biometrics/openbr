/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_NULL_PAINT_DEVICE_H
#define QWT_NULL_PAINT_DEVICE_H 1

#include "qwt_global.h"
#include <qpaintdevice.h>
#include <qpaintengine.h>

/*!
  \brief A null paint device doing nothing

  Sometimes important layout/rendering geometries are not 
  available or changable from the public Qt class interface. 
  ( f.e hidden in the style implementation ).

  QwtNullPaintDevice can be used to manipulate or filter out 
  these informations by analyzing the stream of paint primitives.

  F.e. QwtNullPaintDevice is used by QwtPlotCanvas to identify
  styled backgrounds with rounded corners.
*/

class QWT_EXPORT QwtNullPaintDevice: public QPaintDevice
{
public:
    QwtNullPaintDevice( QPaintEngine::PaintEngineFeatures );
    QwtNullPaintDevice( const QSize &size,
        QPaintEngine::PaintEngineFeatures );

    virtual ~QwtNullPaintDevice();

    void setSize( const QSize &);
    QSize size() const;

    virtual QPaintEngine *paintEngine() const;
    virtual int metric( PaintDeviceMetric metric ) const;

    virtual void drawRects(const QRect *, int );
    virtual void drawRects(const QRectF *, int );

    virtual void drawLines(const QLine *, int );
    virtual void drawLines(const QLineF *, int );

    virtual void drawEllipse(const QRectF &);
    virtual void drawEllipse(const QRect &);

    virtual void drawPath(const QPainterPath &);

    virtual void drawPoints(const QPointF *, int );
    virtual void drawPoints(const QPoint *, int );

    virtual void drawPolygon(
        const QPointF *, int , QPaintEngine::PolygonDrawMode );

    virtual void drawPolygon(
        const QPoint *, int , QPaintEngine::PolygonDrawMode );

    virtual void drawPixmap(const QRectF &,
        const QPixmap &, const QRectF &);

    virtual void drawTextItem(const QPointF &, const QTextItem &);

    virtual void drawTiledPixmap(const QRectF &,
        const QPixmap &, const QPointF &s);

    virtual void drawImage(const QRectF &,
        const QImage &, const QRectF &, Qt::ImageConversionFlags );

    virtual void updateState( const QPaintEngineState &state );

private:
    void init( QPaintEngine::PaintEngineFeatures );

    class PaintEngine;
    PaintEngine *d_engine;

    class PrivateData;
    PrivateData *d_data;
};

#endif
