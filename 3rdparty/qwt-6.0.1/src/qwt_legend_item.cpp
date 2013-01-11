/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_legend_item.h"
#include "qwt_math.h"
#include "qwt_painter.h"
#include "qwt_symbol.h"
#include <qpainter.h>
#include <qdrawutil.h>
#include <qstyle.h>
#include <qpen.h>
#include <qevent.h>
#include <qstyleoption.h>
#include <qapplication.h>

static const int ButtonFrame = 2;
static const int Margin = 2;

static QSize buttonShift( const QwtLegendItem *w )
{
    QStyleOption option;
    option.init( w );

    const int ph = w->style()->pixelMetric(
        QStyle::PM_ButtonShiftHorizontal, &option, w );
    const int pv = w->style()->pixelMetric(
        QStyle::PM_ButtonShiftVertical, &option, w );
    return QSize( ph, pv );
}

class QwtLegendItem::PrivateData
{
public:
    PrivateData():
        itemMode( QwtLegend::ReadOnlyItem ),
        isDown( false ),
        identifierSize( 8, 8 ),
        spacing( Margin )
    {
    }

    QwtLegend::LegendItemMode itemMode;
    bool isDown;

    QSize identifierSize;
    QPixmap identifier;

    int spacing;
};

/*!
  \param parent Parent widget
*/
QwtLegendItem::QwtLegendItem( QWidget *parent ):
    QwtTextLabel( parent )
{
    d_data = new PrivateData;
    setMargin( Margin );
    setIndent( Margin + d_data->identifierSize.width() + 2 * d_data->spacing );
}

//! Destructor
QwtLegendItem::~QwtLegendItem()
{
    delete d_data;
    d_data = NULL;
}

/*!
   Set the text to the legend item

   \param text Text label
    \sa QwtTextLabel::text()
*/
void QwtLegendItem::setText( const QwtText &text )
{
    const int flags = Qt::AlignLeft | Qt::AlignVCenter
        | Qt::TextExpandTabs | Qt::TextWordWrap;

    QwtText txt = text;
    txt.setRenderFlags( flags );

    QwtTextLabel::setText( txt );
}

/*!
   Set the item mode
   The default is QwtLegend::ReadOnlyItem

   \param mode Item mode
   \sa itemMode()
*/
void QwtLegendItem::setItemMode( QwtLegend::LegendItemMode mode )
{
    if ( mode != d_data->itemMode )
    {
        d_data->itemMode = mode;
        d_data->isDown = false;

        setFocusPolicy( mode != QwtLegend::ReadOnlyItem ? Qt::TabFocus : Qt::NoFocus );
        setMargin( ButtonFrame + Margin );

        updateGeometry();
    }
}

/*!
   Return the item mode

   \sa setItemMode()
*/
QwtLegend::LegendItemMode QwtLegendItem::itemMode() const
{
    return d_data->itemMode;
}

/*!
  Assign the identifier
  The identifier needs to be created according to the identifierWidth()

  \param identifier Pixmap representing a plot item

  \sa identifier(), identifierWidth()
*/
void QwtLegendItem::setIdentifier( const QPixmap &identifier )
{
    d_data->identifier = identifier;
    update();
}

/*!
  \return pixmap representing a plot item
  \sa setIdentifier()
*/
QPixmap QwtLegendItem::identifier() const
{
    return d_data->identifier;
}

/*!
  Set the size for the identifier
  Default is 8x8 pixels

  \param size New size

  \sa identifierSize()
*/
void QwtLegendItem::setIdentifierSize( const QSize &size )
{
    QSize sz = size.expandedTo( QSize( 0, 0 ) );
    if ( sz != d_data->identifierSize )
    {
        d_data->identifierSize = sz;
        setIndent( margin() + d_data->identifierSize.width()
            + 2 * d_data->spacing );
        updateGeometry();
    }
}
/*!
   Return the width of the identifier

   \sa setIdentifierSize()
*/
QSize QwtLegendItem::identifierSize() const
{
    return d_data->identifierSize;
}

/*!
   Change the spacing
   \param spacing Spacing
   \sa spacing(), identifierWidth(), QwtTextLabel::margin()
*/
void QwtLegendItem::setSpacing( int spacing )
{
    spacing = qMax( spacing, 0 );
    if ( spacing != d_data->spacing )
    {
        d_data->spacing = spacing;
        setIndent( margin() + d_data->identifierSize.width()
            + 2 * d_data->spacing );
    }
}

/*!
   Return the spacing
   \sa setSpacing(), identifierWidth(), QwtTextLabel::margin()
*/
int QwtLegendItem::spacing() const
{
    return d_data->spacing;
}

/*!
    Check/Uncheck a the item

    \param on check/uncheck
    \sa setItemMode()
*/
void QwtLegendItem::setChecked( bool on )
{
    if ( d_data->itemMode == QwtLegend::CheckableItem )
    {
        const bool isBlocked = signalsBlocked();
        blockSignals( true );

        setDown( on );

        blockSignals( isBlocked );
    }
}

//! Return true, if the item is checked
bool QwtLegendItem::isChecked() const
{
    return d_data->itemMode == QwtLegend::CheckableItem && isDown();
}

//! Set the item being down
void QwtLegendItem::setDown( bool down )
{
    if ( down == d_data->isDown )
        return;

    d_data->isDown = down;
    update();

    if ( d_data->itemMode == QwtLegend::ClickableItem )
    {
        if ( d_data->isDown )
            Q_EMIT pressed();
        else
        {
            Q_EMIT released();
            Q_EMIT clicked();
        }
    }

    if ( d_data->itemMode == QwtLegend::CheckableItem )
        Q_EMIT checked( d_data->isDown );
}

//! Return true, if the item is down
bool QwtLegendItem::isDown() const
{
    return d_data->isDown;
}

//! Return a size hint
QSize QwtLegendItem::sizeHint() const
{
    QSize sz = QwtTextLabel::sizeHint();
    sz.setHeight( qMax( sz.height(), d_data->identifier.height() + 4 ) );

    if ( d_data->itemMode != QwtLegend::ReadOnlyItem )
    {
        sz += buttonShift( this );
        sz = sz.expandedTo( QApplication::globalStrut() );
    }

    return sz;
}

//! Paint event
void QwtLegendItem::paintEvent( QPaintEvent *e )
{
    const QRect cr = contentsRect();

    QPainter painter( this );
    painter.setClipRegion( e->region() );

    if ( d_data->isDown )
    {
        qDrawWinButton( &painter, 0, 0, width(), height(),
            palette(), true );
    }

    painter.save();

    if ( d_data->isDown )
    {
        const QSize shiftSize = buttonShift( this );
        painter.translate( shiftSize.width(), shiftSize.height() );
    }

    painter.setClipRect( cr );

    drawContents( &painter );

    if ( !d_data->identifier.isNull() )
    {
        QRect identRect = cr;
        identRect.setX( identRect.x() + margin() );
        if ( d_data->itemMode != QwtLegend::ReadOnlyItem )
            identRect.setX( identRect.x() + ButtonFrame );

        identRect.setSize( d_data->identifier.size() );
        identRect.moveCenter( QPoint( identRect.center().x(), cr.center().y() ) );

        painter.drawPixmap( identRect, d_data->identifier );
    }

    painter.restore();
}

//! Handle mouse press events
void QwtLegendItem::mousePressEvent( QMouseEvent *e )
{
    if ( e->button() == Qt::LeftButton )
    {
        switch ( d_data->itemMode )
        {
            case QwtLegend::ClickableItem:
            {
                setDown( true );
                return;
            }
            case QwtLegend::CheckableItem:
            {
                setDown( !isDown() );
                return;
            }
            default:;
        }
    }
    QwtTextLabel::mousePressEvent( e );
}

//! Handle mouse release events
void QwtLegendItem::mouseReleaseEvent( QMouseEvent *e )
{
    if ( e->button() == Qt::LeftButton )
    {
        switch ( d_data->itemMode )
        {
            case QwtLegend::ClickableItem:
            {
                setDown( false );
                return;
            }
            case QwtLegend::CheckableItem:
            {
                return; // do nothing, but accept
            }
            default:;
        }
    }
    QwtTextLabel::mouseReleaseEvent( e );
}

//! Handle key press events
void QwtLegendItem::keyPressEvent( QKeyEvent *e )
{
    if ( e->key() == Qt::Key_Space )
    {
        switch ( d_data->itemMode )
        {
            case QwtLegend::ClickableItem:
            {
                if ( !e->isAutoRepeat() )
                    setDown( true );
                return;
            }
            case QwtLegend::CheckableItem:
            {
                if ( !e->isAutoRepeat() )
                    setDown( !isDown() );
                return;
            }
            default:;
        }
    }

    QwtTextLabel::keyPressEvent( e );
}

//! Handle key release events
void QwtLegendItem::keyReleaseEvent( QKeyEvent *e )
{
    if ( e->key() == Qt::Key_Space )
    {
        switch ( d_data->itemMode )
        {
            case QwtLegend::ClickableItem:
            {
                if ( !e->isAutoRepeat() )
                    setDown( false );
                return;
            }
            case QwtLegend::CheckableItem:
            {
                return; // do nothing, but accept
            }
            default:;
        }
    }

    QwtTextLabel::keyReleaseEvent( e );
}