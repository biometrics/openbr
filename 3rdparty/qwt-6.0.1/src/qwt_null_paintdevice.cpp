/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_null_paintdevice.h"
#include <qpaintengine.h>
#include <qpixmap.h>

class QwtNullPaintDevice::PrivateData
{
public:
    PrivateData():
        size( 0, 0 )
    {
    }

    QSize size;
};

class QwtNullPaintDevice::PaintEngine: public QPaintEngine
{
public:
    PaintEngine( QPaintEngine::PaintEngineFeatures );

    virtual bool begin( QPaintDevice * );
    virtual bool end();

    virtual Type type () const;
    virtual void updateState(const QPaintEngineState &);

    virtual void drawRects(const QRect *, int );
    virtual void drawRects(const QRectF *, int );

    virtual void drawLines(const QLine *, int );
    virtual void drawLines(const QLineF *, int );

    virtual void drawEllipse(const QRectF &);
    virtual void drawEllipse(const QRect &);

    virtual void drawPath(const QPainterPath &);

    virtual void drawPoints(const QPointF *, int );
    virtual void drawPoints(const QPoint *, int );

    virtual void drawPolygon(const QPointF *, int , PolygonDrawMode );
    virtual void drawPolygon(const QPoint *, int , PolygonDrawMode );

    virtual void drawPixmap(const QRectF &, 
        const QPixmap &, const QRectF &);

    virtual void drawTextItem(const QPointF &, const QTextItem &);
    virtual void drawTiledPixmap(const QRectF &, 
        const QPixmap &, const QPointF &s);
    virtual void drawImage(const QRectF &, 
        const QImage &, const QRectF &, Qt::ImageConversionFlags );

private:
    QwtNullPaintDevice *d_device;
};
    
QwtNullPaintDevice::PaintEngine::PaintEngine( 
        QPaintEngine::PaintEngineFeatures features ):
    QPaintEngine( features ),
    d_device(NULL)
{
}

bool QwtNullPaintDevice::PaintEngine::begin( 
    QPaintDevice *device )
{
    d_device = static_cast<QwtNullPaintDevice *>( device );
    return true;
}

bool QwtNullPaintDevice::PaintEngine::end()
{
    d_device = NULL;
    return true;
}

QPaintEngine::Type 
QwtNullPaintDevice::PaintEngine::type () const
{
    return QPaintEngine::User;
}

void QwtNullPaintDevice::PaintEngine::drawRects(
    const QRect *rects, int rectCount)
{
    if ( d_device )
        d_device->drawRects( rects, rectCount );
}

void QwtNullPaintDevice::PaintEngine::drawRects(
    const QRectF *rects, int rectCount)
{
    if ( d_device )
        d_device->drawRects( rects, rectCount );
}

void QwtNullPaintDevice::PaintEngine::drawLines(
    const QLine *lines, int lineCount)
{
    if ( d_device )
        d_device->drawLines( lines, lineCount );
}

void QwtNullPaintDevice::PaintEngine::drawLines(
    const QLineF *lines, int lineCount)
{
    if ( d_device )
        d_device->drawLines( lines, lineCount );
}

void QwtNullPaintDevice::PaintEngine::drawEllipse(
    const QRectF &rect)
{
    if ( d_device )
        d_device->drawEllipse( rect );
}

void QwtNullPaintDevice::PaintEngine::drawEllipse(
    const QRect &rect)
{
    if ( d_device )
        d_device->drawEllipse( rect );
}


void QwtNullPaintDevice::PaintEngine::drawPath(
    const QPainterPath &path)
{
    if ( d_device )
        d_device->drawPath( path );
}

void QwtNullPaintDevice::PaintEngine::drawPoints(
    const QPointF *points, int pointCount)
{
    if ( d_device )
        d_device->drawPoints( points, pointCount );
}

void QwtNullPaintDevice::PaintEngine::drawPoints(
    const QPoint *points, int pointCount)
{
    if ( d_device )
        d_device->drawPoints( points, pointCount );
}

void QwtNullPaintDevice::PaintEngine::drawPolygon(
    const QPointF *points, int pointCount, PolygonDrawMode mode)
{
    if ( d_device )
        d_device->drawPolygon( points, pointCount, mode );
}

void QwtNullPaintDevice::PaintEngine::drawPolygon(
    const QPoint *points, int pointCount, PolygonDrawMode mode)
{
    if ( d_device )
        d_device->drawPolygon( points, pointCount, mode );
}

void QwtNullPaintDevice::PaintEngine::drawPixmap( 
    const QRectF &rect, const QPixmap &pm, const QRectF &subRect )
{
    if ( d_device )
        d_device->drawPixmap( rect, pm, subRect );
}

void QwtNullPaintDevice::PaintEngine::drawTextItem(
    const QPointF &pos, const QTextItem &textItem)
{
    if ( d_device )
        d_device->drawTextItem( pos, textItem );
}

void QwtNullPaintDevice::PaintEngine::drawTiledPixmap(
    const QRectF &rect, const QPixmap &pixmap, 
    const QPointF &subRect)
{
    if ( d_device )
        d_device->drawTiledPixmap( rect, pixmap, subRect );
}

void QwtNullPaintDevice::PaintEngine::drawImage(
    const QRectF &rect, const QImage &image, 
    const QRectF &subRect, Qt::ImageConversionFlags flags)
{
    if ( d_device )
        d_device->drawImage( rect, image, subRect, flags );
}

void QwtNullPaintDevice::PaintEngine::updateState(
    const QPaintEngineState &state)
{
    if ( d_device )
        d_device->updateState( state );
}

//! Constructor
QwtNullPaintDevice::QwtNullPaintDevice( 
    QPaintEngine::PaintEngineFeatures features )
{
    init( features );
}

//! Constructor
QwtNullPaintDevice::QwtNullPaintDevice( const QSize &size,
    QPaintEngine::PaintEngineFeatures features )
{
    init( features );
    d_data->size = size;
}

void QwtNullPaintDevice::init( 
    QPaintEngine::PaintEngineFeatures features )
{
    d_engine = new PaintEngine( features );
    d_data = new PrivateData;
}

//! Destructor
QwtNullPaintDevice::~QwtNullPaintDevice()
{
    delete d_engine;
    delete d_data;
}

/*!
   Set the size of the paint device

   \param size Size
   \sa size()
*/
void QwtNullPaintDevice::setSize( const QSize & size )
{
    d_data->size = size;
}

/*! 
    \return Size of the paint device
    \sa setSize()
*/
QSize QwtNullPaintDevice::size() const
{
    return d_data->size;
}

//! See QPaintDevice::paintEngine()
QPaintEngine *QwtNullPaintDevice::paintEngine() const
{
    return d_engine;
}

/*! 
    See QPaintDevice::metric()
    \sa setSize()
*/
int QwtNullPaintDevice::metric( PaintDeviceMetric metric ) const
{
    static QPixmap pm;

    int value;

    switch ( metric ) 
    {
        case PdmWidth:
            value = qMax( d_data->size.width(), 0 );
            break;
        case PdmHeight:
            value = qMax( d_data->size.height(), 0 );
            break;
        case PdmNumColors:
            value = 16777216;
            break;
        case PdmDepth:
            value = 24;
            break;
        case PdmPhysicalDpiX:
        case PdmDpiY:
        case PdmPhysicalDpiY:
        case PdmWidthMM:
        case PdmHeightMM:
        case PdmDpiX:
        default:
            value = 0;
    }
    return value;

}

//! See QPaintEngine::drawRects()
void QwtNullPaintDevice::drawRects(
    const QRect *rects, int rectCount)
{
    Q_UNUSED(rects);
    Q_UNUSED(rectCount);
}

//! See QPaintEngine::drawRects()
void QwtNullPaintDevice::drawRects(
    const QRectF *rects, int rectCount)
{
    Q_UNUSED(rects);
    Q_UNUSED(rectCount);
}

//! See QPaintEngine::drawLines()
void QwtNullPaintDevice::drawLines(
    const QLine *lines, int lineCount)
{
    Q_UNUSED(lines);
    Q_UNUSED(lineCount);
}

//! See QPaintEngine::drawLines()
void QwtNullPaintDevice::drawLines(
    const QLineF *lines, int lineCount)
{
    Q_UNUSED(lines);
    Q_UNUSED(lineCount);
}

//! See QPaintEngine::drawEllipse()
void QwtNullPaintDevice::drawEllipse( const QRectF &rect )
{
    Q_UNUSED(rect);
}

//! See QPaintEngine::drawEllipse()
void QwtNullPaintDevice::drawEllipse( const QRect &rect )
{
    Q_UNUSED(rect);
}

//! See QPaintEngine::drawPath()
void QwtNullPaintDevice::drawPath( const QPainterPath &path )
{
    Q_UNUSED(path);
}

//! See QPaintEngine::drawPoints()
void QwtNullPaintDevice::drawPoints(
    const QPointF *points, int pointCount)
{
    Q_UNUSED(points);
    Q_UNUSED(pointCount);
}

//! See QPaintEngine::drawPoints()
void QwtNullPaintDevice::drawPoints(
    const QPoint *points, int pointCount)
{
    Q_UNUSED(points);
    Q_UNUSED(pointCount);
}

//! See QPaintEngine::drawPolygon()
void QwtNullPaintDevice::drawPolygon(
    const QPointF *points, int pointCount, 
    QPaintEngine::PolygonDrawMode mode)
{
    Q_UNUSED(points);
    Q_UNUSED(pointCount);
    Q_UNUSED(mode);
}

//! See QPaintEngine::drawPolygon()
void QwtNullPaintDevice::drawPolygon(
    const QPoint *points, int pointCount, 
    QPaintEngine::PolygonDrawMode mode)
{
    Q_UNUSED(points);
    Q_UNUSED(pointCount);
    Q_UNUSED(mode);
}

//! See QPaintEngine::drawPixmap()
void QwtNullPaintDevice::drawPixmap( const QRectF &rect, 
    const QPixmap &pm, const QRectF &subRect )
{
    Q_UNUSED(rect);
    Q_UNUSED(pm);
    Q_UNUSED(subRect);
}

//! See QPaintEngine::drawTextItem()
void QwtNullPaintDevice::drawTextItem(
    const QPointF &pos, const QTextItem &textItem)
{
    Q_UNUSED(pos);
    Q_UNUSED(textItem);
}

//! See QPaintEngine::drawTiledPixmap()
void QwtNullPaintDevice::drawTiledPixmap(
    const QRectF &rect, const QPixmap &pixmap, 
    const QPointF &subRect)
{
    Q_UNUSED(rect);
    Q_UNUSED(pixmap);
    Q_UNUSED(subRect);
}

//! See QPaintEngine::drawImage()
void QwtNullPaintDevice::drawImage(
    const QRectF &rect, const QImage &image, 
    const QRectF &subRect, Qt::ImageConversionFlags flags)
{
    Q_UNUSED(rect);
    Q_UNUSED(image);
    Q_UNUSED(subRect);
    Q_UNUSED(flags);
}

//! See QPaintEngine::updateState()
void QwtNullPaintDevice::updateState( 
    const QPaintEngineState &state )
{
    Q_UNUSED(state);
}
