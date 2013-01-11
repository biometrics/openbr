/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_plot_curve.h"
#include "qwt_math.h"
#include "qwt_clipper.h"
#include "qwt_painter.h"
#include "qwt_legend.h"
#include "qwt_legend_item.h"
#include "qwt_scale_map.h"
#include "qwt_plot.h"
#include "qwt_plot_canvas.h"
#include "qwt_curve_fitter.h"
#include "qwt_symbol.h"
#include <qpainter.h>
#include <qpixmap.h>
#include <qalgorithms.h>
#include <qmath.h>

static int verifyRange( int size, int &i1, int &i2 )
{
    if ( size < 1 )
        return 0;

    i1 = qBound( 0, i1, size - 1 );
    i2 = qBound( 0, i2, size - 1 );

    if ( i1 > i2 )
        qSwap( i1, i2 );

    return ( i2 - i1 + 1 );
}

class QwtPlotCurve::PrivateData
{
public:
    PrivateData():
        style( QwtPlotCurve::Lines ),
        baseline( 0.0 ),
        symbol( NULL ),
        attributes( 0 ),
        paintAttributes( QwtPlotCurve::ClipPolygons ),
        legendAttributes( 0 )
    {
        pen = QPen( Qt::black );
        curveFitter = new QwtSplineCurveFitter;
    }

    ~PrivateData()
    {
        delete symbol;
        delete curveFitter;
    }

    QwtPlotCurve::CurveStyle style;
    double baseline;

    const QwtSymbol *symbol;
    QwtCurveFitter *curveFitter;

    QPen pen;
    QBrush brush;

    QwtPlotCurve::CurveAttributes attributes;
    QwtPlotCurve::PaintAttributes paintAttributes;

    QwtPlotCurve::LegendAttributes legendAttributes;
};

/*!
  Constructor
  \param title Title of the curve
*/
QwtPlotCurve::QwtPlotCurve( const QwtText &title ):
    QwtPlotSeriesItem<QPointF>( title )
{
    init();
}

/*!
  Constructor
  \param title Title of the curve
*/
QwtPlotCurve::QwtPlotCurve( const QString &title ):
    QwtPlotSeriesItem<QPointF>( QwtText( title ) )
{
    init();
}

//! Destructor
QwtPlotCurve::~QwtPlotCurve()
{
    delete d_data;
}

//! Initialize internal members
void QwtPlotCurve::init()
{
    setItemAttribute( QwtPlotItem::Legend );
    setItemAttribute( QwtPlotItem::AutoScale );

    d_data = new PrivateData;
    d_series = new QwtPointSeriesData();

    setZ( 20.0 );
}

//! \return QwtPlotItem::Rtti_PlotCurve
int QwtPlotCurve::rtti() const
{
    return QwtPlotItem::Rtti_PlotCurve;
}

/*!
  Specify an attribute how to draw the curve

  \param attribute Paint attribute
  \param on On/Off
  \sa testPaintAttribute()
*/
void QwtPlotCurve::setPaintAttribute( PaintAttribute attribute, bool on )
{
    if ( on )
        d_data->paintAttributes |= attribute;
    else
        d_data->paintAttributes &= ~attribute;
}

/*!
    \brief Return the current paint attributes
    \sa setPaintAttribute()
*/
bool QwtPlotCurve::testPaintAttribute( PaintAttribute attribute ) const
{
    return ( d_data->paintAttributes & attribute );
}

/*!
  Specify an attribute how to draw the legend identifier

  \param attribute Attribute
  \param on On/Off
  /sa testLegendAttribute()
*/
void QwtPlotCurve::setLegendAttribute( LegendAttribute attribute, bool on )
{
    if ( on )
        d_data->legendAttributes |= attribute;
    else
        d_data->legendAttributes &= ~attribute;
}

/*!
    \brief Return the current paint attributes
    \sa setLegendAttribute()
*/
bool QwtPlotCurve::testLegendAttribute( LegendAttribute attribute ) const
{
    return ( d_data->legendAttributes & attribute );
}

/*!
  Set the curve's drawing style

  \param style Curve style
  \sa style()
*/
void QwtPlotCurve::setStyle( CurveStyle style )
{
    if ( style != d_data->style )
    {
        d_data->style = style;
        itemChanged();
    }
}

/*!
    Return the current style
    \sa setStyle()
*/
QwtPlotCurve::CurveStyle QwtPlotCurve::style() const
{
    return d_data->style;
}

/*!
  Assign a symbol

  \param symbol Symbol
  \sa symbol()
*/
void QwtPlotCurve::setSymbol( const QwtSymbol *symbol )
{
    if ( symbol != d_data->symbol )
    {
        delete d_data->symbol;
        d_data->symbol = symbol;
        itemChanged();
    }
}

/*!
  \return Current symbol or NULL, when no symbol has been assigned
  \sa setSymbol()
*/
const QwtSymbol *QwtPlotCurve::symbol() const
{
    return d_data->symbol;
}

/*!
  Assign a pen

  \param pen New pen
  \sa pen(), brush()
*/
void QwtPlotCurve::setPen( const QPen &pen )
{
    if ( pen != d_data->pen )
    {
        d_data->pen = pen;
        itemChanged();
    }
}

/*!
  \return Pen used to draw the lines
  \sa setPen(), brush()
*/
const QPen& QwtPlotCurve::pen() const
{
    return d_data->pen;
}

/*!
  \brief Assign a brush.

   In case of brush.style() != QBrush::NoBrush
   and style() != QwtPlotCurve::Sticks
   the area between the curve and the baseline will be filled.

   In case !brush.color().isValid() the area will be filled by
   pen.color(). The fill algorithm simply connects the first and the
   last curve point to the baseline. So the curve data has to be sorted
   (ascending or descending).

  \param brush New brush
  \sa brush(), setBaseline(), baseline()
*/
void QwtPlotCurve::setBrush( const QBrush &brush )
{
    if ( brush != d_data->brush )
    {
        d_data->brush = brush;
        itemChanged();
    }
}

/*!
  \return Brush used to fill the area between lines and the baseline
  \sa setBrush(), setBaseline(), baseline()
*/
const QBrush& QwtPlotCurve::brush() const
{
    return d_data->brush;
}

/*!
  Draw an interval of the curve

  \param painter Painter
  \param xMap Maps x-values into pixel coordinates.
  \param yMap Maps y-values into pixel coordinates.
  \param canvasRect Contents rect of the canvas
  \param from Index of the first point to be painted
  \param to Index of the last point to be painted. If to < 0 the
         curve will be painted to its last point.

  \sa drawCurve(), drawSymbols(),
*/
void QwtPlotCurve::drawSeries( QPainter *painter,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    const QRectF &canvasRect, int from, int to ) const
{
    if ( !painter || dataSize() <= 0 )
        return;

    if ( to < 0 )
        to = dataSize() - 1;

    if ( verifyRange( dataSize(), from, to ) > 0 )
    {
        painter->save();
        painter->setPen( d_data->pen );

        /*
          Qt 4.0.0 is slow when drawing lines, but it's even
          slower when the painter has a brush. So we don't
          set the brush before we really need it.
         */

        drawCurve( painter, d_data->style, xMap, yMap, canvasRect, from, to );
        painter->restore();

        if ( d_data->symbol &&
            ( d_data->symbol->style() != QwtSymbol::NoSymbol ) )
        {
            painter->save();
            drawSymbols( painter, *d_data->symbol,
                xMap, yMap, canvasRect, from, to );
            painter->restore();
        }
    }
}

/*!
  \brief Draw the line part (without symbols) of a curve interval.
  \param painter Painter
  \param style curve style, see QwtPlotCurve::CurveStyle
  \param xMap x map
  \param yMap y map
  \param canvasRect Contents rect of the canvas
  \param from index of the first point to be painted
  \param to index of the last point to be painted
  \sa draw(), drawDots(), drawLines(), drawSteps(), drawSticks()
*/
void QwtPlotCurve::drawCurve( QPainter *painter, int style,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    const QRectF &canvasRect, int from, int to ) const
{
    switch ( style )
    {
        case Lines:
            if ( testCurveAttribute( Fitted ) )
            {
                // we always need the complete
                // curve for fitting
                from = 0;
                to = dataSize() - 1;
            }
            drawLines( painter, xMap, yMap, canvasRect, from, to );
            break;
        case Sticks:
            drawSticks( painter, xMap, yMap, canvasRect, from, to );
            break;
        case Steps:
            drawSteps( painter, xMap, yMap, canvasRect, from, to );
            break;
        case Dots:
            drawDots( painter, xMap, yMap, canvasRect, from, to );
            break;
        case NoCurve:
        default:
            break;
    }
}

/*!
  \brief Draw lines

  If the CurveAttribute Fitted is enabled a QwtCurveFitter tries
  to interpolate/smooth the curve, before it is painted.

  \param painter Painter
  \param xMap x map
  \param yMap y map
  \param canvasRect Contents rect of the canvas
  \param from index of the first point to be painted
  \param to index of the last point to be painted

  \sa setCurveAttribute(), setCurveFitter(), draw(),
      drawLines(), drawDots(), drawSteps(), drawSticks()
*/
void QwtPlotCurve::drawLines( QPainter *painter,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    const QRectF &canvasRect, int from, int to ) const
{
    int size = to - from + 1;
    if ( size <= 0 )
        return;

    const bool doAlign = QwtPainter::roundingAlignment( painter );

    QPolygonF polyline( size );

    QPointF *points = polyline.data();
    for ( int i = from; i <= to; i++ )
    {
        const QPointF sample = d_series->sample( i );

        double x = xMap.transform( sample.x() );
        double y = yMap.transform( sample.y() );
        if ( doAlign )
        {
            x = qRound( x );
            y = qRound( y );
        }

        points[i - from].rx() = x;
        points[i - from].ry() = y;
    }

    if ( ( d_data->attributes & Fitted ) && d_data->curveFitter )
        polyline = d_data->curveFitter->fitCurve( polyline );

    if ( d_data->paintAttributes & ClipPolygons )
    {
        qreal pw = qMax( qreal( 1.0 ), painter->pen().widthF());
        const QPolygonF clipped = QwtClipper::clipPolygonF( 
            canvasRect.adjusted(-pw, -pw, pw, pw), polyline, false );

        QwtPainter::drawPolyline( painter, clipped );
    }
    else
    {
        QwtPainter::drawPolyline( painter, polyline );
    }

    if ( d_data->brush.style() != Qt::NoBrush )
        fillCurve( painter, xMap, yMap, canvasRect, polyline );
}

/*!
  Draw sticks

  \param painter Painter
  \param xMap x map
  \param yMap y map
  \param canvasRect Contents rect of the canvas
  \param from index of the first point to be painted
  \param to index of the last point to be painted

  \sa draw(), drawCurve(), drawDots(), drawLines(), drawSteps()
*/
void QwtPlotCurve::drawSticks( QPainter *painter,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    const QRectF &, int from, int to ) const
{
    painter->save();
    painter->setRenderHint( QPainter::Antialiasing, false );

    const bool doAlign = QwtPainter::roundingAlignment( painter );

    double x0 = xMap.transform( d_data->baseline );
    double y0 = yMap.transform( d_data->baseline );
    if ( doAlign )
    {
        x0 = qRound( x0 );
        y0 = qRound( y0 );
    }

    const Qt::Orientation o = orientation();

    for ( int i = from; i <= to; i++ )
    {
        const QPointF sample = d_series->sample( i );
        double xi = xMap.transform( sample.x() );
        double yi = yMap.transform( sample.y() );
        if ( doAlign )
        {
            xi = qRound( xi );
            yi = qRound( yi );
        }

        if ( o == Qt::Horizontal )
            QwtPainter::drawLine( painter, x0, yi, xi, yi );
        else
            QwtPainter::drawLine( painter, xi, y0, xi, yi );
    }

    painter->restore();
}

/*!
  Draw dots

  \param painter Painter
  \param xMap x map
  \param yMap y map
  \param canvasRect Contents rect of the canvas
  \param from index of the first point to be painted
  \param to index of the last point to be painted

  \sa draw(), drawCurve(), drawSticks(), drawLines(), drawSteps()
*/
void QwtPlotCurve::drawDots( QPainter *painter,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    const QRectF &canvasRect, int from, int to ) const
{
    const bool doFill = d_data->brush.style() != Qt::NoBrush;
    const bool doAlign = QwtPainter::roundingAlignment( painter );

    QPolygonF polyline;
    if ( doFill )
        polyline.resize( to - from + 1 );

    QPointF *points = polyline.data();

    for ( int i = from; i <= to; i++ )
    {
        const QPointF sample = d_series->sample( i );
        double xi = xMap.transform( sample.x() );
        double yi = yMap.transform( sample.y() );
        if ( doAlign )
        {
            xi = qRound( xi );
            yi = qRound( yi );
        }

        QwtPainter::drawPoint( painter, QPointF( xi, yi ) );

        if ( doFill )
        {
            points[i - from].rx() = xi;
            points[i - from].ry() = yi;
        }
    }

    if ( doFill )
        fillCurve( painter, xMap, yMap, canvasRect, polyline );
}

/*!
  Draw step function

  The direction of the steps depends on Inverted attribute.

  \param painter Painter
  \param xMap x map
  \param yMap y map
  \param canvasRect Contents rect of the canvas
  \param from index of the first point to be painted
  \param to index of the last point to be painted

  \sa CurveAttribute, setCurveAttribute(),
      draw(), drawCurve(), drawDots(), drawLines(), drawSticks()
*/
void QwtPlotCurve::drawSteps( QPainter *painter,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    const QRectF &canvasRect, int from, int to ) const
{
    const bool doAlign = QwtPainter::roundingAlignment( painter );

    QPolygonF polygon( 2 * ( to - from ) + 1 );
    QPointF *points = polygon.data();

    bool inverted = orientation() == Qt::Vertical;
    if ( d_data->attributes & Inverted )
        inverted = !inverted;

    int i, ip;
    for ( i = from, ip = 0; i <= to; i++, ip += 2 )
    {
        const QPointF sample = d_series->sample( i );
        double xi = xMap.transform( sample.x() );
        double yi = yMap.transform( sample.y() );
        if ( doAlign )
        {
            xi = qRound( xi );
            yi = qRound( yi );
        }

        if ( ip > 0 )
        {
            const QPointF &p0 = points[ip - 2];
            QPointF &p = points[ip - 1];

            if ( inverted )
            {
                p.rx() = p0.x();
                p.ry() = yi;
            }
            else
            {
                p.rx() = xi;
                p.ry() = p0.y();
            }
        }

        points[ip].rx() = xi;
        points[ip].ry() = yi;
    }

    if ( d_data->paintAttributes & ClipPolygons )
    {
        const QPolygonF clipped = QwtClipper::clipPolygonF( 
            canvasRect, polygon, false );

        QwtPainter::drawPolyline( painter, clipped );
    }
    else
    {
        QwtPainter::drawPolyline( painter, polygon );
    }

    if ( d_data->brush.style() != Qt::NoBrush )
        fillCurve( painter, xMap, yMap, canvasRect, polygon );
}


/*!
  Specify an attribute for drawing the curve

  \param attribute Curve attribute
  \param on On/Off

  /sa testCurveAttribute(), setCurveFitter()
*/
void QwtPlotCurve::setCurveAttribute( CurveAttribute attribute, bool on )
{
    if ( bool( d_data->attributes & attribute ) == on )
        return;

    if ( on )
        d_data->attributes |= attribute;
    else
        d_data->attributes &= ~attribute;

    itemChanged();
}

/*!
    \return true, if attribute is enabled
    \sa setCurveAttribute()
*/
bool QwtPlotCurve::testCurveAttribute( CurveAttribute attribute ) const
{
    return d_data->attributes & attribute;
}

/*!
  Assign a curve fitter

  The curve fitter "smooths" the curve points, when the Fitted
  CurveAttribute is set. setCurveFitter(NULL) also disables curve fitting.

  The curve fitter operates on the translated points ( = widget coordinates)
  to be functional for logarithmic scales. Obviously this is less performant
  for fitting algorithms, that reduce the number of points.

  For situations, where curve fitting is used to improve the performance
  of painting huge series of points it might be better to execute the fitter
  on the curve points once and to cache the result in the QwtSeriesData object.

  \param curveFitter() Curve fitter
  \sa Fitted
*/
void QwtPlotCurve::setCurveFitter( QwtCurveFitter *curveFitter )
{
    delete d_data->curveFitter;
    d_data->curveFitter = curveFitter;

    itemChanged();
}

/*!
  Get the curve fitter. If curve fitting is disabled NULL is returned.

  \return Curve fitter
  \sa setCurveFitter(), Fitted
*/
QwtCurveFitter *QwtPlotCurve::curveFitter() const
{
    return d_data->curveFitter;
}

/*!
  Fill the area between the curve and the baseline with
  the curve brush

  \param painter Painter
  \param xMap x map
  \param yMap y map
  \param canvasRect Contents rect of the canvas
  \param polygon Polygon - will be modified !

  \sa setBrush(), setBaseline(), setStyle()
*/
void QwtPlotCurve::fillCurve( QPainter *painter,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    const QRectF &canvasRect, QPolygonF &polygon ) const
{
    if ( d_data->brush.style() == Qt::NoBrush )
        return;

    closePolyline( painter, xMap, yMap, polygon );
    if ( polygon.count() <= 2 ) // a line can't be filled
        return;

    QBrush brush = d_data->brush;
    if ( !brush.color().isValid() )
        brush.setColor( d_data->pen.color() );

    if ( d_data->paintAttributes & ClipPolygons )
        polygon = QwtClipper::clipPolygonF( canvasRect, polygon, true );

    painter->save();

    painter->setPen( Qt::NoPen );
    painter->setBrush( brush );

    QwtPainter::drawPolygon( painter, polygon );

    painter->restore();
}

/*!
  \brief Complete a polygon to be a closed polygon including the 
         area between the original polygon and the baseline.

  \param painter Painter
  \param xMap X map
  \param yMap Y map
  \param polygon Polygon to be completed
*/
void QwtPlotCurve::closePolyline( QPainter *painter,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    QPolygonF &polygon ) const
{
    if ( polygon.size() < 2 )
        return;

    const bool doAlign = QwtPainter::roundingAlignment( painter );

    double baseline = d_data->baseline;
    
    if ( orientation() == Qt::Vertical )
    {
        if ( yMap.transformation()->type() == QwtScaleTransformation::Log10 )
        {
            if ( baseline < QwtScaleMap::LogMin )
                baseline = QwtScaleMap::LogMin;
        }

        double refY = yMap.transform( baseline );
        if ( doAlign )
            refY = qRound( refY );

        polygon += QPointF( polygon.last().x(), refY );
        polygon += QPointF( polygon.first().x(), refY );
    }
    else
    {
        if ( xMap.transformation()->type() == QwtScaleTransformation::Log10 )
        {
            if ( baseline < QwtScaleMap::LogMin )
                baseline = QwtScaleMap::LogMin;
        }

        double refX = xMap.transform( baseline );
        if ( doAlign )
            refX = qRound( refX );

        polygon += QPointF( refX, polygon.last().y() );
        polygon += QPointF( refX, polygon.first().y() );
    }
}

/*!
  Draw symbols

  \param painter Painter
  \param symbol Curve symbol
  \param xMap x map
  \param yMap y map
  \param canvasRect Contents rect of the canvas
  \param from Index of the first point to be painted
  \param to Index of the last point to be painted

  \sa setSymbol(), drawSeries(), drawCurve()
*/
void QwtPlotCurve::drawSymbols( QPainter *painter, const QwtSymbol &symbol,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    const QRectF &canvasRect, int from, int to ) const
{
    const bool doAlign = QwtPainter::roundingAlignment( painter );

    bool usePixmap = testPaintAttribute( CacheSymbols );
    if ( usePixmap && !doAlign )
    {
        // Don't use the pixmap, when the paint device
        // could generate scalable vectors

        usePixmap = false;
    }

    if ( usePixmap )
    {
        QPixmap pm( symbol.boundingSize() );
        pm.fill( Qt::transparent );

        const double pw2 = 0.5 * pm.width();
        const double ph2 = 0.5 * pm.height();

        QPainter p( &pm );
        p.setRenderHints( painter->renderHints() );
        symbol.drawSymbol( &p, QPointF( pw2, ph2 ) );
        p.end();

        for ( int i = from; i <= to; i++ )
        {
            const QPointF sample = d_series->sample( i );

            double xi = xMap.transform( sample.x() );
            double yi = yMap.transform( sample.y() );
            if ( doAlign )
            {
                xi = qRound( xi );
                yi = qRound( yi );
            }

            if ( canvasRect.contains( xi, yi ) )
            {
                const int left = qCeil( xi ) - pw2;
                const int top = qCeil( yi ) - ph2;

                painter->drawPixmap( left, top, pm );
            }
        }
    }
    else
    {
        const int chunkSize = 500;

        for ( int i = from; i <= to; i += chunkSize )
        {
            const int n = qMin( chunkSize, to - i + 1 );

            QPolygonF points;
            for ( int j = 0; j < n; j++ )
            {
                const QPointF sample = d_series->sample( i + j );

                const double xi = xMap.transform( sample.x() );
                const double yi = yMap.transform( sample.y() );

                if ( canvasRect.contains( xi, yi ) )
                    points += QPointF( xi, yi );
            }

            if ( points.size() > 0 )
                symbol.drawSymbols( painter, points );
        }
    }
}

/*!
  \brief Set the value of the baseline

  The baseline is needed for filling the curve with a brush or
  the Sticks drawing style.
  The interpretation of the baseline depends on the CurveType.
  With QwtPlotCurve::Yfx, the baseline is interpreted as a horizontal line
  at y = baseline(), with QwtPlotCurve::Yfy, it is interpreted as a vertical
  line at x = baseline().

  The default value is 0.0.

  \param value Value of the baseline
  \sa baseline(), setBrush(), setStyle(), setStyle()
*/
void QwtPlotCurve::setBaseline( double value )
{
    if ( d_data->baseline != value )
    {
        d_data->baseline = value;
        itemChanged();
    }
}

/*!
  \return Value of the baseline
  \sa setBaseline()
*/
double QwtPlotCurve::baseline() const
{
    return d_data->baseline;
}

/*!
  Find the closest curve point for a specific position

  \param pos Position, where to look for the closest curve point
  \param dist If dist != NULL, closestPoint() returns the distance between
              the position and the clostest curve point
  \return Index of the closest curve point, or -1 if none can be found
          ( f.e when the curve has no points )
  \note closestPoint() implements a dumb algorithm, that iterates
        over all points
*/
int QwtPlotCurve::closestPoint( const QPoint &pos, double *dist ) const
{
    if ( plot() == NULL || dataSize() <= 0 )
        return -1;

    const QwtScaleMap xMap = plot()->canvasMap( xAxis() );
    const QwtScaleMap yMap = plot()->canvasMap( yAxis() );

    int index = -1;
    double dmin = 1.0e10;

    for ( uint i = 0; i < dataSize(); i++ )
    {
        const QPointF sample = d_series->sample( i );

        const double cx = xMap.transform( sample.x() ) - pos.x();
        const double cy = yMap.transform( sample.y() ) - pos.y();

        const double f = qwtSqr( cx ) + qwtSqr( cy );
        if ( f < dmin )
        {
            index = i;
            dmin = f;
        }
    }
    if ( dist )
        *dist = qSqrt( dmin );

    return index;
}

/*!
   \brief Update the widget that represents the item on the legend

   \param legend Legend
   \sa drawLegendIdentifier(), legendItem(), QwtPlotItem::Legend
*/
void QwtPlotCurve::updateLegend( QwtLegend *legend ) const
{
    if ( legend && testItemAttribute( QwtPlotItem::Legend )
        && ( d_data->legendAttributes & QwtPlotCurve::LegendShowSymbol )
        && d_data->symbol
        && d_data->symbol->style() != QwtSymbol::NoSymbol )
    {
        QWidget *lgdItem = legend->find( this );
        if ( lgdItem == NULL )
        {
            lgdItem = legendItem();
            if ( lgdItem )
                legend->insert( this, lgdItem );
        }

        QwtLegendItem *l = qobject_cast<QwtLegendItem *>( lgdItem );
        if ( l )
        {
            QSize sz = d_data->symbol->boundingSize();
            sz += QSize( 2, 2 ); // margin

            if ( d_data->legendAttributes & QwtPlotCurve::LegendShowLine )
            {
                // Avoid, that the line is completely covered by the symbol

                int w = qCeil( 1.5 * sz.width() );
                if ( w % 2 )
                    w++;

                sz.setWidth( qMax( 8, w ) );
            }

            l->setIdentifierSize( sz );
        }
    }

    QwtPlotItem::updateLegend( legend );
}

/*!
  \brief Draw the identifier representing the curve on the legend

  \param painter Painter
  \param rect Bounding rectangle for the identifier

  \sa setLegendAttribute(), QwtPlotItem::Legend
*/
void QwtPlotCurve::drawLegendIdentifier(
    QPainter *painter, const QRectF &rect ) const
{
    if ( rect.isEmpty() )
        return;

    const int dim = qMin( rect.width(), rect.height() );

    QSize size( dim, dim );

    QRectF r( 0, 0, size.width(), size.height() );
    r.moveCenter( rect.center() );

    if ( d_data->legendAttributes == 0 )
    {
        QBrush brush = d_data->brush;
        if ( brush.style() == Qt::NoBrush )
        {
            if ( style() != QwtPlotCurve::NoCurve )
                brush = QBrush( pen().color() );
            else if ( d_data->symbol &&
                ( d_data->symbol->style() != QwtSymbol::NoSymbol ) )
            {
                brush = QBrush( d_data->symbol->pen().color() );
            }
        }
        if ( brush.style() != Qt::NoBrush )
            painter->fillRect( r, brush );
    }
    if ( d_data->legendAttributes & QwtPlotCurve::LegendShowBrush )
    {
        if ( d_data->brush.style() != Qt::NoBrush )
            painter->fillRect( r, d_data->brush );
    }
    if ( d_data->legendAttributes & QwtPlotCurve::LegendShowLine )
    {
        if ( pen() != Qt::NoPen )
        {
            painter->setPen( pen() );
            QwtPainter::drawLine( painter, rect.left(), rect.center().y(),
                                  rect.right() - 1.0, rect.center().y() );
        }
    }
    if ( d_data->legendAttributes & QwtPlotCurve::LegendShowSymbol )
    {
        if ( d_data->symbol &&
            ( d_data->symbol->style() != QwtSymbol::NoSymbol ) )
        {
            QSize symbolSize = d_data->symbol->boundingSize();
            symbolSize -= QSize( 2, 2 );

            // scale the symbol size down if it doesn't fit into rect.

            double xRatio = 1.0;
            if ( rect.width() < symbolSize.width() )
                xRatio = rect.width() / symbolSize.width();
            double yRatio = 1.0;
            if ( rect.height() < symbolSize.height() )
                yRatio = rect.height() / symbolSize.height();

            const double ratio = qMin( xRatio, yRatio );

            painter->save();
            painter->scale( ratio, ratio );

            d_data->symbol->drawSymbol( painter, rect.center() / ratio );

            painter->restore();
        }
    }
}

/*!
  Initialize data with an array of points (explicitly shared).

  \param samples Vector of points
*/
void QwtPlotCurve::setSamples( const QVector<QPointF> &samples )
{
    delete d_series;
    d_series = new QwtPointSeriesData( samples );
    itemChanged();
}

#ifndef QWT_NO_COMPAT

/*!
  \brief Initialize the data by pointing to memory blocks which 
         are not managed by QwtPlotCurve.

  setRawSamples is provided for efficiency. 
  It is important to keep the pointers
  during the lifetime of the underlying QwtCPointerData class.

  \param xData pointer to x data
  \param yData pointer to y data
  \param size size of x and y

  \sa QwtCPointerData
*/
void QwtPlotCurve::setRawSamples( 
    const double *xData, const double *yData, int size )
{
    delete d_series;
    d_series = new QwtCPointerData( xData, yData, size );
    itemChanged();
}

/*!
  Set data by copying x- and y-values from specified memory blocks.
  Contrary to setRawSamples(), this function makes a 'deep copy' of
  the data.

  \param xData pointer to x values
  \param yData pointer to y values
  \param size size of xData and yData

  \sa QwtPointArrayData
*/
void QwtPlotCurve::setSamples( 
    const double *xData, const double *yData, int size )
{
    delete d_series;
    d_series = new QwtPointArrayData( xData, yData, size );
    itemChanged();
}

/*!
  \brief Initialize data with x- and y-arrays (explicitly shared)

  \param xData x data
  \param yData y data

  \sa QwtPointArrayData
*/
void QwtPlotCurve::setSamples( const QVector<double> &xData,
    const QVector<double> &yData )
{
    delete d_series;
    d_series = new QwtPointArrayData( xData, yData );
    itemChanged();
}
#endif // !QWT_NO_COMPAT

