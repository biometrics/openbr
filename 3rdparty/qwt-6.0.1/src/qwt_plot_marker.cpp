/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_plot_marker.h"
#include "qwt_painter.h"
#include "qwt_scale_map.h"
#include "qwt_symbol.h"
#include "qwt_text.h"
#include "qwt_math.h"
#include "qwt_legend.h"
#include "qwt_legend_item.h"
#include <qpainter.h>

class QwtPlotMarker::PrivateData
{
public:
    PrivateData():
        labelAlignment( Qt::AlignCenter ),
        labelOrientation( Qt::Horizontal ),
        spacing( 2 ),
        symbol( NULL ),
        style( QwtPlotMarker::NoLine ),
        xValue( 0.0 ),
        yValue( 0.0 )
    {
    }

    ~PrivateData()
    {
        delete symbol;
    }

    QwtText label;
    Qt::Alignment labelAlignment;
    Qt::Orientation labelOrientation;
    int spacing;

    QPen pen;
    const QwtSymbol *symbol;
    LineStyle style;

    double xValue;
    double yValue;
};

//! Sets alignment to Qt::AlignCenter, and style to QwtPlotMarker::NoLine
QwtPlotMarker::QwtPlotMarker():
    QwtPlotItem( QwtText( "Marker" ) )
{
    d_data = new PrivateData;
    setZ( 30.0 );
}

//! Destructor
QwtPlotMarker::~QwtPlotMarker()
{
    delete d_data;
}

//! \return QwtPlotItem::Rtti_PlotMarker
int QwtPlotMarker::rtti() const
{
    return QwtPlotItem::Rtti_PlotMarker;
}

//! Return Value
QPointF QwtPlotMarker::value() const
{
    return QPointF( d_data->xValue, d_data->yValue );
}

//! Return x Value
double QwtPlotMarker::xValue() const
{
    return d_data->xValue;
}

//! Return y Value
double QwtPlotMarker::yValue() const
{
    return d_data->yValue;
}

//! Set Value
void QwtPlotMarker::setValue( const QPointF& pos )
{
    setValue( pos.x(), pos.y() );
}

//! Set Value
void QwtPlotMarker::setValue( double x, double y )
{
    if ( x != d_data->xValue || y != d_data->yValue )
    {
        d_data->xValue = x;
        d_data->yValue = y;
        itemChanged();
    }
}

//! Set X Value
void QwtPlotMarker::setXValue( double x )
{
    setValue( x, d_data->yValue );
}

//! Set Y Value
void QwtPlotMarker::setYValue( double y )
{
    setValue( d_data->xValue, y );
}

/*!
  Draw the marker

  \param painter Painter
  \param xMap x Scale Map
  \param yMap y Scale Map
  \param canvasRect Contents rect of the canvas in painter coordinates
*/
void QwtPlotMarker::draw( QPainter *painter,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    const QRectF &canvasRect ) const
{
    const QPointF pos( xMap.transform( d_data->xValue ), 
        yMap.transform( d_data->yValue ) );

    // draw lines

    drawLines( painter, canvasRect, pos );

    // draw symbol
    if ( d_data->symbol &&
        ( d_data->symbol->style() != QwtSymbol::NoSymbol ) )
    {
        d_data->symbol->drawSymbol( painter, pos );
    }

    drawLabel( painter, canvasRect, pos );
}

/*!
  Draw the lines marker

  \param painter Painter
  \param canvasRect Contents rect of the canvas in painter coordinates
  \param pos Position of the marker, translated into widget coordinates

  \sa drawLabel(), QwtSymbol::drawSymbol()
*/
void QwtPlotMarker::drawLines( QPainter *painter,
    const QRectF &canvasRect, const QPointF &pos ) const
{
    if ( d_data->style == NoLine )
        return;

    const bool doAlign = QwtPainter::roundingAlignment( painter );

    painter->setPen( d_data->pen );
    if ( d_data->style == QwtPlotMarker::HLine ||
        d_data->style == QwtPlotMarker::Cross )
    {
        double y = pos.y();
        if ( doAlign )
            y = qRound( y );

        QwtPainter::drawLine( painter, canvasRect.left(),
            y, canvasRect.right() - 1.0, y );
    }
    if ( d_data->style == QwtPlotMarker::VLine ||
        d_data->style == QwtPlotMarker::Cross )
    {
        double x = pos.x();
        if ( doAlign )
            x = qRound( x );

        QwtPainter::drawLine( painter, x,
            canvasRect.top(), x, canvasRect.bottom() - 1.0 );
    }
}

/*!
  Align and draw the text label of the marker

  \param painter Painter
  \param canvasRect Contents rect of the canvas in painter coordinates
  \param pos Position of the marker, translated into widget coordinates

  \sa drawLabel(), QwtSymbol::drawSymbol()
*/
void QwtPlotMarker::drawLabel( QPainter *painter,
    const QRectF &canvasRect, const QPointF &pos ) const
{
    if ( d_data->label.isEmpty() )
        return;

    Qt::Alignment align = d_data->labelAlignment;
    QPointF alignPos = pos;

    QSizeF symbolOff( 0, 0 );

    switch ( d_data->style )
    {
        case QwtPlotMarker::VLine:
        {
            // In VLine-style the y-position is pointless and
            // the alignment flags are relative to the canvas

            if ( d_data->labelAlignment & Qt::AlignTop )
            {
                alignPos.setY( canvasRect.top() );
                align &= ~Qt::AlignTop;
                align |= Qt::AlignBottom;
            }
            else if ( d_data->labelAlignment & Qt::AlignBottom )
            {
                // In HLine-style the x-position is pointless and
                // the alignment flags are relative to the canvas

                alignPos.setY( canvasRect.bottom() - 1 );
                align &= ~Qt::AlignBottom;
                align |= Qt::AlignTop;
            }
            else
            {
                alignPos.setY( canvasRect.center().y() );
            }
            break;
        }
        case QwtPlotMarker::HLine:
        {
            if ( d_data->labelAlignment & Qt::AlignLeft )
            {
                alignPos.setX( canvasRect.left() );
                align &= ~Qt::AlignLeft;
                align |= Qt::AlignRight;
            }
            else if ( d_data->labelAlignment & Qt::AlignRight )
            {
                alignPos.setX( canvasRect.right() - 1 );
                align &= ~Qt::AlignRight;
                align |= Qt::AlignLeft;
            }
            else
            {
                alignPos.setX( canvasRect.center().x() );
            }
            break;
        }
        default:
        {
            if ( d_data->symbol &&
                ( d_data->symbol->style() != QwtSymbol::NoSymbol ) )
            {
                symbolOff = d_data->symbol->size() + QSizeF( 1, 1 );
                symbolOff /= 2;
            }
        }
    }

    qreal pw2 = d_data->pen.widthF() / 2.0;
    if ( pw2 == 0.0 )
        pw2 = 0.5;

    const int spacing = d_data->spacing;

    const qreal xOff = qMax( pw2, symbolOff.width() );
    const qreal yOff = qMax( pw2, symbolOff.height() );

    const QSizeF textSize = d_data->label.textSize( painter->font() );

    if ( align & Qt::AlignLeft )
    {
        alignPos.rx() -= xOff + spacing;
        if ( d_data->labelOrientation == Qt::Vertical )
            alignPos.rx() -= textSize.height();
        else
            alignPos.rx() -= textSize.width();
    }
    else if ( align & Qt::AlignRight )
    {
        alignPos.rx() += xOff + spacing;
    }
    else
    {
        if ( d_data->labelOrientation == Qt::Vertical )
            alignPos.rx() -= textSize.height() / 2;
        else
            alignPos.rx() -= textSize.width() / 2;
    }

    if ( align & Qt::AlignTop )
    {
        alignPos.ry() -= yOff + spacing;
        if ( d_data->labelOrientation != Qt::Vertical )
            alignPos.ry() -= textSize.height();
    }
    else if ( align & Qt::AlignBottom )
    {
        alignPos.ry() += yOff + spacing;
        if ( d_data->labelOrientation == Qt::Vertical )
            alignPos.ry() += textSize.width();
    }
    else
    {
        if ( d_data->labelOrientation == Qt::Vertical )
            alignPos.ry() += textSize.width() / 2;
        else
            alignPos.ry() -= textSize.height() / 2;
    }

    painter->translate( alignPos.x(), alignPos.y() );
    if ( d_data->labelOrientation == Qt::Vertical )
        painter->rotate( -90.0 );

    const QRectF textRect( 0, 0, textSize.width(), textSize.height() );
    d_data->label.draw( painter, textRect );
}

/*!
  \brief Set the line style
  \param style Line style. 
  \sa lineStyle()
*/
void QwtPlotMarker::setLineStyle( LineStyle style )
{
    if ( style != d_data->style )
    {
        d_data->style = style;
        itemChanged();
    }
}

/*!
  \return the line style
  \sa setLineStyle()
*/
QwtPlotMarker::LineStyle QwtPlotMarker::lineStyle() const
{
    return d_data->style;
}

/*!
  \brief Assign a symbol
  \param symbol New symbol
  \sa symbol()
*/
void QwtPlotMarker::setSymbol( const QwtSymbol *symbol )
{
    if ( symbol != d_data->symbol )
    {
        delete d_data->symbol;
        d_data->symbol = symbol;
        itemChanged();
    }
}

/*!
  \return the symbol
  \sa setSymbol(), QwtSymbol
*/
const QwtSymbol *QwtPlotMarker::symbol() const
{
    return d_data->symbol;
}

/*!
  \brief Set the label
  \param label label text
  \sa label()
*/
void QwtPlotMarker::setLabel( const QwtText& label )
{
    if ( label != d_data->label )
    {
        d_data->label = label;
        itemChanged();
    }
}

/*!
  \return the label
  \sa setLabel()
*/
QwtText QwtPlotMarker::label() const
{
    return d_data->label;
}

/*!
  \brief Set the alignment of the label

  In case of QwtPlotMarker::HLine the alignment is relative to the
  y position of the marker, but the horizontal flags correspond to the
  canvas rectangle. In case of QwtPlotMarker::VLine the alignment is
  relative to the x position of the marker, but the vertical flags
  correspond to the canvas rectangle.

  In all other styles the alignment is relative to the marker's position.

  \param align Alignment. 
  \sa labelAlignment(), labelOrientation()
*/
void QwtPlotMarker::setLabelAlignment( Qt::Alignment align )
{
    if ( align != d_data->labelAlignment )
    {
        d_data->labelAlignment = align;
        itemChanged();
    }
}

/*!
  \return the label alignment
  \sa setLabelAlignment(), setLabelOrientation()
*/
Qt::Alignment QwtPlotMarker::labelAlignment() const
{
    return d_data->labelAlignment;
}

/*!
  \brief Set the orientation of the label

  When orientation is Qt::Vertical the label is rotated by 90.0 degrees
  ( from bottom to top ).

  \param orientation Orientation of the label

  \sa labelOrientation(), setLabelAlignment()
*/
void QwtPlotMarker::setLabelOrientation( Qt::Orientation orientation )
{
    if ( orientation != d_data->labelOrientation )
    {
        d_data->labelOrientation = orientation;
        itemChanged();
    }
}

/*!
  \return the label orientation
  \sa setLabelOrientation(), labelAlignment()
*/
Qt::Orientation QwtPlotMarker::labelOrientation() const
{
    return d_data->labelOrientation;
}

/*!
  \brief Set the spacing

  When the label is not centered on the marker position, the spacing
  is the distance between the position and the label.

  \param spacing Spacing
  \sa spacing(), setLabelAlignment()
*/
void QwtPlotMarker::setSpacing( int spacing )
{
    if ( spacing < 0 )
        spacing = 0;

    if ( spacing == d_data->spacing )
        return;

    d_data->spacing = spacing;
    itemChanged();
}

/*!
  \return the spacing
  \sa setSpacing()
*/
int QwtPlotMarker::spacing() const
{
    return d_data->spacing;
}

/*!
  Specify a pen for the line.

  \param pen New pen
  \sa linePen()
*/
void QwtPlotMarker::setLinePen( const QPen &pen )
{
    if ( pen != d_data->pen )
    {
        d_data->pen = pen;
        itemChanged();
    }
}

/*!
  \return the line pen
  \sa setLinePen()
*/
const QPen &QwtPlotMarker::linePen() const
{
    return d_data->pen;
}

QRectF QwtPlotMarker::boundingRect() const
{
    return QRectF( d_data->xValue, d_data->yValue, 0.0, 0.0 );
}

/*!
   \brief Update the widget that represents the item on the legend

   \param legend Legend
   \sa drawLegendIdentifier(), legendItem(), itemChanged(), QwtLegend()

   \note In the default setting QwtPlotItem::Legend is disabled 
*/
void QwtPlotMarker::updateLegend( QwtLegend *legend ) const
{
    if ( legend && testItemAttribute( QwtPlotItem::Legend )
        && d_data->symbol && d_data->symbol->style() != QwtSymbol::NoSymbol )
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
            l->setIdentifierSize( d_data->symbol->boundingSize() );
    }

    QwtPlotItem::updateLegend( legend );
}

/*!
  \brief Draw the identifier representing the marker on the legend

  \param painter Painter
  \param rect Bounding rectangle for the identifier

  \sa updateLegend(), QwtPlotItem::Legend
*/
void QwtPlotMarker::drawLegendIdentifier(
    QPainter *painter, const QRectF &rect ) const
{
    if ( rect.isEmpty() )
        return;

    painter->save();
    painter->setClipRect( rect, Qt::IntersectClip );

    if ( d_data->style != QwtPlotMarker::NoLine )
    {
        painter->setPen( d_data->pen );

        if ( d_data->style == QwtPlotMarker::HLine ||
            d_data->style == QwtPlotMarker::Cross )
        {
            QwtPainter::drawLine( painter, rect.left(), rect.center().y(),
                rect.right(), rect.center().y() );
        }

        if ( d_data->style == QwtPlotMarker::VLine ||
            d_data->style == QwtPlotMarker::Cross )
        {
            QwtPainter::drawLine( painter, rect.center().x(), rect.top(),
                rect.center().x(), rect.bottom() );
        }
    }

    if ( d_data->symbol && d_data->symbol->style() != QwtSymbol::NoSymbol )
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

        painter->scale( ratio, ratio );
        d_data->symbol->drawSymbol( painter, rect.center() / ratio );
    }

    painter->restore();
}

