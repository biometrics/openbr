/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_plot_textlabel.h"
#include "qwt_painter.h"
#include "qwt_scale_map.h"
#include <qpainter.h>
#include <qpixmap.h>

static QRect qwtItemRect( int renderFlags,
    const QRect &rect, const QSize &itemSize ) 
{
    int x;
    if ( renderFlags & Qt::AlignLeft )
    {
        x = rect.left();
    }
    else if ( renderFlags & Qt::AlignRight )
    {
        x = rect.right() - itemSize.width();
    }
    else
    {
        x = rect.center().x() - 0.5 * itemSize.width();
    }

    int y;
    if ( renderFlags & Qt::AlignTop ) 
    {
        y = rect.top();
    }
    else if ( renderFlags & Qt::AlignBottom )
    {
        y = rect.bottom() - itemSize.width();
    }
    else
    {
        y = rect.center().y() - 0.5 * itemSize.height();
    }

    return QRect( x, y, itemSize.width(), itemSize.height() );
}

class QwtPlotTextLabel::PrivateData
{   
public:
    PrivateData():
        margin( 5 )
    {
    }

    QwtText text;
    int margin;

    QPixmap pixmap;
};  

/*!
   \brief Constructor

   Initializes an text label with an empty text

   Sets the following item attributes:

   - QwtPlotItem::AutoScale: true
   - QwtPlotItem::Legend:    false

   The z value is initialized by 150

   \sa QwtPlotItem::setItemAttribute(), QwtPlotItem::setZ()
*/

QwtPlotTextLabel::QwtPlotTextLabel():
    QwtPlotItem( QwtText( "Label" ) )
{
    d_data = new PrivateData;

    setItemAttribute( QwtPlotItem::AutoScale, false );
    setItemAttribute( QwtPlotItem::Legend, false );

    setZ( 150 );
}

//! Destructor
QwtPlotTextLabel::~QwtPlotTextLabel()
{
    delete d_data;
}

//! \return QwtPlotItem::Rtti_PlotTextLabel
int QwtPlotTextLabel::rtti() const
{
    return QwtPlotItem::Rtti_PlotTextLabel;
}

/*!
  Set the text 

  The label will be aligned to the plot canvas according to
  the alignment flags of text.

  \param text Text to be displayed

  \sa text(), QwtText::renderFlags()
*/
void QwtPlotTextLabel::setText( const QwtText &text )
{
    if ( d_data->text != text )
    {
        d_data->text = text;

        invalidateCache();
        itemChanged();
    }
}

/*!
  \return Text to be displayed
  \sa setText()
*/
QwtText QwtPlotTextLabel::text() const
{
    return d_data->text;
}

/*!
  Set the margin

  The margin is the distance between the contentsRect()
  of the plot canvas and the rectangle where the label can
  be displayed

  \param margin Margin

  \sa margin()
 */
void QwtPlotTextLabel::setMargin( int margin )
{
    margin = qMax( margin, 0 );
    if ( d_data->margin != margin )
    {
        d_data->margin = margin;
        itemChanged();
    }
}

/*!
  \return Margin added to the contentsMargins() of the canvas
  \sa setMargin()
*/
int QwtPlotTextLabel::margin() const
{
    return d_data->margin;
}

/*!
  Draw the text label

  \param painter Painter
  \param xMap x Scale Map
  \param yMap y Scale Map
  \param canvasRect Contents rectangle of the canvas in painter coordinates
*/

void QwtPlotTextLabel::draw( QPainter *painter,
    const QwtScaleMap &xMap, const QwtScaleMap &yMap,
    const QRectF &canvasRect ) const
{
    Q_UNUSED( xMap );
    Q_UNUSED( yMap );

    const int m = d_data->margin;

    const QRectF adjustedRect = canvasRect.adjusted( m, m, -m, -m );

    const bool doAlign = QwtPainter::roundingAlignment( painter );

    if ( doAlign )
    {
        // when the paint device is aligning it is not one
        // where scalability matters ( PDF, SVG ).
        // As rendering a text label is an expensive operation
        // we use a cache.

        const QSize sz = d_data->text.textSize( painter->font() ).toSize();

        if ( d_data->pixmap.isNull() || sz != d_data->pixmap.size()  )
        {
            d_data->pixmap = QPixmap( sz );
            d_data->pixmap.fill( Qt::transparent );

            const QRect cacheRect( QPoint(), sz );

            QPainter pmPainter( &d_data->pixmap );
            d_data->text.draw( &pmPainter, cacheRect );
        }

        const QRect r = qwtItemRect( d_data->text.renderFlags(),
            adjustedRect.toRect(), d_data->pixmap.size() );

        painter->drawPixmap( r, d_data->pixmap );
    }
    else
    {
        d_data->text.draw( painter, adjustedRect );
    }
}

//!  Invalidate all internal cache
void QwtPlotTextLabel::invalidateCache()
{
    d_data->pixmap = QPixmap();
}
