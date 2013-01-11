/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_slider.h"
#include "qwt_painter.h"
#include "qwt_scale_draw.h"
#include "qwt_scale_map.h"
#include <qevent.h>
#include <qdrawutil.h>
#include <qpainter.h>
#include <qalgorithms.h>
#include <qmath.h>
#include <qstyle.h>
#include <qstyleoption.h>
#include <qapplication.h>

class QwtSlider::PrivateData
{
public:
    QRect sliderRect;

    QSize handleSize;
    int borderWidth;
    int spacing;

    QwtSlider::ScalePos scalePos;
    QwtSlider::BackgroundStyles bgStyle;

    /*
      Scale and values might have different maps. This is
      confusing and I can't see strong arguments for such
      a feature. TODO ...
     */
    QwtScaleMap map; // linear map
    mutable QSize sizeHintCache;
};

/*!
  \brief Constructor
  \param parent parent widget
  \param orientation Orientation of the slider. Can be Qt::Horizontal
         or Qt::Vertical. Defaults to Qt::Horizontal.
  \param scalePos Position of the scale.
         Defaults to QwtSlider::NoScale.
  \param bgStyle Background style. QwtSlider::Trough draws the
         slider button in a trough, QwtSlider::Slot draws
         a slot underneath the button. An or-combination of both
         may also be used. The default is QwtSlider::Trough.

  QwtSlider enforces valid combinations of its orientation and scale position.
  If the combination is invalid, the scale position will be set to NoScale.
  Valid combinations are:
  - Qt::Horizonal with NoScale, TopScale, or BottomScale;
  - Qt::Vertical with NoScale, LeftScale, or RightScale.
*/
QwtSlider::QwtSlider( QWidget *parent,
        Qt::Orientation orientation, ScalePos scalePos, 
        BackgroundStyles bgStyle ):
    QwtAbstractSlider( orientation, parent )
{
    initSlider( orientation, scalePos, bgStyle );
}

void QwtSlider::initSlider( Qt::Orientation orientation,
    ScalePos scalePos, BackgroundStyles bgStyle )
{
    if ( orientation == Qt::Vertical )
        setSizePolicy( QSizePolicy::Fixed, QSizePolicy::Expanding );
    else
        setSizePolicy( QSizePolicy::Expanding, QSizePolicy::Fixed );

    setAttribute( Qt::WA_WState_OwnSizePolicy, false );

    d_data = new QwtSlider::PrivateData;

    d_data->borderWidth = 2;
    d_data->spacing = 4;
    d_data->scalePos = scalePos;
    d_data->bgStyle = bgStyle;

    const int handleThickness = 16;
    d_data->handleSize.setWidth( 2 * handleThickness );
    d_data->handleSize.setHeight( handleThickness );

    if ( !( bgStyle & QwtSlider::Trough ) )
        d_data->handleSize.transpose();

    if ( orientation == Qt::Vertical )
        d_data->handleSize.transpose();

    d_data->sliderRect.setRect( 0, 0, 8, 8 );

    QwtScaleDraw::Alignment align;
    if ( orientation == Qt::Vertical )
    {
        // enforce a valid combination of scale position and orientation
        if ( ( d_data->scalePos == QwtSlider::BottomScale ) 
            || ( d_data->scalePos == QwtSlider::TopScale ) )
        {
            d_data->scalePos = NoScale;
        }

        // adopt the policy of layoutSlider (NoScale lays out like Left)
        if ( d_data->scalePos == QwtSlider::RightScale )
            align = QwtScaleDraw::RightScale;
        else
            align = QwtScaleDraw::LeftScale;
    }
    else
    {
        // enforce a valid combination of scale position and orientation
        if ( ( d_data->scalePos == QwtSlider::LeftScale ) 
            || ( d_data->scalePos == QwtSlider::RightScale ) )
        {
            d_data->scalePos = QwtSlider::NoScale;
        }

        // adopt the policy of layoutSlider (NoScale lays out like Bottom)
        if ( d_data->scalePos == QwtSlider::TopScale )
            align = QwtScaleDraw::TopScale;
        else
            align = QwtScaleDraw::BottomScale;
    }

    scaleDraw()->setAlignment( align );
    scaleDraw()->setLength( 100 );

    setRange( 0.0, 100.0, 1.0 );
    setValue( 0.0 );
}

QwtSlider::~QwtSlider()
{
    delete d_data;
}

/*!
  \brief Set the orientation.
  \param o Orientation. Allowed values are Qt::Horizontal and Qt::Vertical.

  If the new orientation and the old scale position are an invalid combination,
  the scale position will be set to QwtSlider::NoScale.
  \sa QwtAbstractSlider::orientation()
*/
void QwtSlider::setOrientation( Qt::Orientation o )
{
    if ( o == orientation() )
        return;

    if ( o == Qt::Horizontal )
    {
        if ( d_data->scalePos == LeftScale 
            || d_data->scalePos == RightScale )
        {
            d_data->scalePos = NoScale;
        }
    }
    else // if (o == Qt::Vertical)
    {
        if ( d_data->scalePos == BottomScale || 
            d_data->scalePos == TopScale )
        {
            d_data->scalePos = NoScale;
        }
    }

    if ( !testAttribute( Qt::WA_WState_OwnSizePolicy ) )
    {
        QSizePolicy sp = sizePolicy();
        sp.transpose();
        setSizePolicy( sp );

        setAttribute( Qt::WA_WState_OwnSizePolicy, false );
    }

    QwtAbstractSlider::setOrientation( o );
    layoutSlider( true );
}

/*!
  \brief Change the scale position (and slider orientation).

  \param scalePos Position of the scale.

  A valid combination of scale position and orientation is enforced:
  - if the new scale position is Left or Right, the scale orientation will
    become Qt::Vertical;
  - if the new scale position is Bottom or Top the scale orientation will
    become Qt::Horizontal;
  - if the new scale position is QwtSlider::NoScale, the scale
    orientation will not change.
*/
void QwtSlider::setScalePosition( ScalePos scalePos )
{
    if ( d_data->scalePos == scalePos )
        return;

    d_data->scalePos = scalePos;

    switch ( d_data->scalePos )
    {
        case QwtSlider::BottomScale:
        {
            setOrientation( Qt::Horizontal );
            scaleDraw()->setAlignment( QwtScaleDraw::BottomScale );
            break;
        }
        case QwtSlider::TopScale:
        {
            setOrientation( Qt::Horizontal );
            scaleDraw()->setAlignment( QwtScaleDraw::TopScale );
            break;
        }
        case QwtSlider::LeftScale:
        {
            setOrientation( Qt::Vertical );
            scaleDraw()->setAlignment( QwtScaleDraw::LeftScale );
            break;
        }
        case RightScale:
        {
            QwtSlider::setOrientation( Qt::Vertical );
            scaleDraw()->setAlignment( QwtScaleDraw::RightScale );
            break;
        }
        default:
        {
            // nothing
        }
    }

    layoutSlider( true );
}

//! Return the scale position.
QwtSlider::ScalePos QwtSlider::scalePosition() const
{
    return d_data->scalePos;
}

/*!
  \brief Change the slider's border width
  \param width Border width
*/
void QwtSlider::setBorderWidth( int width )
{
    if ( width < 0 )
        width = 0;

    if ( width != d_data->borderWidth )
    {
        d_data->borderWidth = width;
        layoutSlider( true );
    }
}

/*!
  \return the border width.
  \sa setBorderWidth()
*/
int QwtSlider::borderWidth() const
{
    return d_data->borderWidth;
}

/*!
  \brief Change the spacing between pipe and scale

  A spacing of 0 means, that the backbone of the scale is below
  the trough.

  The default setting is 4 pixels.

  \param spacing Number of pixels
  \sa spacing();
*/
void QwtSlider::setSpacing( int spacing )
{
    if ( spacing <= 0 )
        spacing = 0;

    if ( spacing != d_data->spacing  )
    {
        d_data->spacing = spacing;
        layoutSlider( true );
    }
}

/*!
  \return Number of pixels between slider and scale
  \sa setSpacing()
*/
int QwtSlider::spacing() const
{
    return d_data->spacing;
}

/*!
  \brief Set the slider's handle size
  \param width Width
  \param height Height

  \sa handleSize()
*/
void QwtSlider::setHandleSize( int width, int height )
{
    setHandleSize( QSize( width, height ) );
}

/*!
  \brief Set the slider's handle size
  \param size New size

  \sa handleSize()
*/
void QwtSlider::setHandleSize( const QSize &size )
{
    const QSize handleSize = size.expandedTo( QSize( 8, 4 ) );
    if ( handleSize != d_data->handleSize )
    {
        d_data->handleSize = handleSize;
        layoutSlider( true );
    }
}

/*!
  \return Size of the handle.
  \sa setHandleSize()
*/
QSize QwtSlider::handleSize() const
{
    return d_data->handleSize;
}

/*!
  \brief Set a scale draw

  For changing the labels of the scales, it
  is necessary to derive from QwtScaleDraw and
  overload QwtScaleDraw::label().

  \param scaleDraw ScaleDraw object, that has to be created with
                   new and will be deleted in ~QwtSlider or the next
                   call of setScaleDraw().

  \sa scaleDraw()
*/
void QwtSlider::setScaleDraw( QwtScaleDraw *scaleDraw )
{
    const QwtScaleDraw *previousScaleDraw = this->scaleDraw();
    if ( scaleDraw == NULL || scaleDraw == previousScaleDraw )
        return;

    if ( previousScaleDraw )
        scaleDraw->setAlignment( previousScaleDraw->alignment() );

    setAbstractScaleDraw( scaleDraw );
    layoutSlider( true );
}

/*!
  \return the scale draw of the slider
  \sa setScaleDraw()
*/
const QwtScaleDraw *QwtSlider::scaleDraw() const
{
    return static_cast<const QwtScaleDraw *>( abstractScaleDraw() );
}

/*!
  \return the scale draw of the slider
  \sa setScaleDraw()
*/
QwtScaleDraw *QwtSlider::scaleDraw()
{
    return static_cast<QwtScaleDraw *>( abstractScaleDraw() );
}

//! Notify changed scale
void QwtSlider::scaleChange()
{
    layoutSlider( true );
}

/*!
   Draw the slider into the specified rectangle.

   \param painter Painter
   \param sliderRect Bounding rectangle of the slider
*/
void QwtSlider::drawSlider( 
    QPainter *painter, const QRect &sliderRect ) const
{
    QRect innerRect( sliderRect );

    if ( d_data->bgStyle & QwtSlider::Trough )
    {
        const int bw = d_data->borderWidth;

        qDrawShadePanel( painter, sliderRect, palette(), true, bw, NULL );

        innerRect = sliderRect.adjusted( bw, bw, -bw, -bw );
        painter->fillRect( innerRect, palette().brush( QPalette::Mid ) );
    }

    if ( d_data->bgStyle & QwtSlider::Groove )
    {
        int ws = 4;
        int ds = d_data->handleSize.width() / 2 - 4;
        if ( ds < 1 )
            ds = 1;

        QRect rSlot;
        if ( orientation() == Qt::Horizontal )
        {
            if ( innerRect.height() & 1 )
                ws++;

            rSlot = QRect( innerRect.x() + ds,
                    innerRect.y() + ( innerRect.height() - ws ) / 2,
                    innerRect.width() - 2 * ds, ws );
        }
        else
        {
            if ( innerRect.width() & 1 )
                ws++;

            rSlot = QRect( innerRect.x() + ( innerRect.width() - ws ) / 2,
                           innerRect.y() + ds,
                           ws, innerRect.height() - 2 * ds );
        }

        QBrush brush = palette().brush( QPalette::Dark );
        qDrawShadePanel( painter, rSlot, palette(), true, 1 , &brush );
    }

    if ( isValid() )
        drawHandle( painter, innerRect, transform( value() ) );
}

/*!
  Draw the thumb at a position

  \param painter Painter
  \param sliderRect Bounding rectangle of the slider
  \param pos Position of the slider thumb
*/
void QwtSlider::drawHandle( QPainter *painter, 
    const QRect &sliderRect, int pos ) const
{
    const int bw = d_data->borderWidth;

    pos++; // shade line points one pixel below
    if ( orientation() == Qt::Horizontal )
    {
        QRect handleRect(
            pos - d_data->handleSize.width() / 2,
            sliderRect.y(), 
            d_data->handleSize.width(), 
            sliderRect.height()
        );

        qDrawShadePanel( painter, 
            handleRect, palette(), false, bw,
            &palette().brush( QPalette::Button ) );

        qDrawShadeLine( painter, pos, sliderRect.top() + bw,
            pos, sliderRect.bottom() - bw,
            palette(), true, 1 );
    }
    else // Vertical
    {
        QRect handleRect(
            sliderRect.left(), 
            pos - d_data->handleSize.height() / 2,
            sliderRect.width(), 
            d_data->handleSize.height()
        );

        qDrawShadePanel( painter, 
            handleRect, palette(), false, bw,
            &palette().brush( QPalette::Button ) );

        qDrawShadeLine( painter, sliderRect.left() + bw, pos,
            sliderRect.right() - bw, pos,
            palette(), true, 1 );
    }
}

/*!
   Map and round a value into widget coordinates
   \param value Value
*/
int QwtSlider::transform( double value ) const
{
    return qRound( d_data->map.transform( value ) );
}

/*!
   Determine the value corresponding to a specified mouse location.
   \param pos Mouse position
*/
double QwtSlider::getValue( const QPoint &pos )
{
    return d_data->map.invTransform(
        orientation() == Qt::Horizontal ? pos.x() : pos.y() );
}

/*!
  \brief Determine scrolling mode and direction
  \param p point
  \param scrollMode Scrolling mode
  \param direction Direction
*/
void QwtSlider::getScrollMode( const QPoint &p,
    QwtAbstractSlider::ScrollMode &scrollMode, int &direction ) const
{
    if ( !d_data->sliderRect.contains( p ) )
    {
        scrollMode = QwtAbstractSlider::ScrNone;
        direction = 0;
        return;
    }

    const int pos = ( orientation() == Qt::Horizontal ) ? p.x() : p.y();
    const int markerPos = transform( value() );

    if ( ( pos > markerPos - d_data->handleSize.width() / 2 )
        && ( pos < markerPos + d_data->handleSize.width() / 2 ) )
    {
        scrollMode = QwtAbstractSlider::ScrMouse;
        direction = 0;
        return;
    }

    scrollMode = QwtAbstractSlider::ScrPage;
    direction = ( pos > markerPos ) ? 1 : -1;

    if ( scaleDraw()->scaleMap().p1() > scaleDraw()->scaleMap().p2() )
        direction = -direction;
}

/*!
   Qt paint event
   \param event Paint event
*/
void QwtSlider::paintEvent( QPaintEvent *event )
{
    QPainter painter( this );
    painter.setClipRegion( event->region() );

    QStyleOption opt;
    opt.init(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);

    if ( d_data->scalePos != QwtSlider::NoScale )
    {
        if ( !d_data->sliderRect.contains( event->rect() ) )
            scaleDraw()->draw( &painter, palette() );
    }

    drawSlider( &painter, d_data->sliderRect );

    if ( hasFocus() )
        QwtPainter::drawFocusRect( &painter, this, d_data->sliderRect );
}

//! Qt resize event
void QwtSlider::resizeEvent( QResizeEvent * )
{
    layoutSlider( false );
}

//! Qt change event handler
void QwtSlider::changeEvent( QEvent *event )
{
    switch( event->type() )
    {
        case QEvent::StyleChange:
        case QEvent::FontChange:
            layoutSlider( true );
            break;
        default: 
            break;
    }
}

/*!
  Recalculate the slider's geometry and layout based on
  the current rect and fonts.
  \param update_geometry  notify the layout system and call update
         to redraw the scale
*/
void QwtSlider::layoutSlider( bool update_geometry )
{
    int handleThickness;
    if ( orientation() == Qt::Horizontal )
        handleThickness = d_data->handleSize.width();
    else
        handleThickness = d_data->handleSize.height();

    int sld1 = handleThickness / 2 - 1;
    int sld2 = handleThickness / 2 + handleThickness % 2;

    if ( d_data->bgStyle & QwtSlider::Trough )
    {
        sld1 += d_data->borderWidth;
        sld2 += d_data->borderWidth;
    }

    int scd = 0;
    if ( d_data->scalePos != QwtSlider::NoScale )
    {
        int d1, d2;
        scaleDraw()->getBorderDistHint( font(), d1, d2 );
        scd = qMax( d1, d2 );
    }

    int slo = scd - sld1;
    if ( slo < 0 )
        slo = 0;

    int x, y, length;
    QRect sliderRect;

    length = x = y = 0;

    const QRect cr = contentsRect();
    if ( orientation() == Qt::Horizontal )
    {
        int sh = d_data->handleSize.height();
        if ( d_data->bgStyle & QwtSlider::Trough )
            sh += 2 * d_data->borderWidth;

        sliderRect.setLeft( cr.left() + slo );
        sliderRect.setRight( cr.right() - slo );
        sliderRect.setTop( cr.top() );
        sliderRect.setBottom( cr.top() + sh - 1);

        if ( d_data->scalePos == QwtSlider::BottomScale )
        {
            y = sliderRect.bottom() + d_data->spacing;
        }
        else if ( d_data->scalePos == QwtSlider::TopScale )
        {
            sliderRect.setTop( cr.bottom() - sh + 1 );
            sliderRect.setBottom( cr.bottom() );

            y = sliderRect.top() - d_data->spacing;
        }

        x = sliderRect.left() + sld1;
        length = sliderRect.width() - ( sld1 + sld2 );
    }
    else // Qt::Vertical
    {
        int sw = d_data->handleSize.width();
        if ( d_data->bgStyle & QwtSlider::Trough )
            sw += 2 * d_data->borderWidth;

        sliderRect.setLeft( cr.right() - sw + 1 );
        sliderRect.setRight( cr.right() );
        sliderRect.setTop( cr.top() + slo );
        sliderRect.setBottom( cr.bottom() - slo );

        if ( d_data->scalePos == QwtSlider::LeftScale )
        {
            x = sliderRect.left() - d_data->spacing;
        }
        else if ( d_data->scalePos == QwtSlider::RightScale )
        {
            sliderRect.setLeft( cr.left() );
            sliderRect.setRight( cr.left() + sw - 1);

            x = sliderRect.right() + d_data->spacing;
        }

        y = sliderRect.top() + sld1;
        length = sliderRect.height() - ( sld1 + sld2 );
    }

    d_data->sliderRect = sliderRect;

    scaleDraw()->move( x, y );
    scaleDraw()->setLength( length );

    d_data->map.setPaintInterval( scaleDraw()->scaleMap().p1(),
        scaleDraw()->scaleMap().p2() );

    if ( update_geometry )
    {
        d_data->sizeHintCache = QSize(); // invalidate
        updateGeometry();
        update();
    }
}

//! Notify change of value
void QwtSlider::valueChange()
{
    QwtAbstractSlider::valueChange();
    update();
}


//! Notify change of range
void QwtSlider::rangeChange()
{
    d_data->map.setScaleInterval( minValue(), maxValue() );

    if ( autoScale() )
        rescale( minValue(), maxValue() );

    QwtAbstractSlider::rangeChange();
    layoutSlider( true );
}

/*!
  Set the background style.
*/
void QwtSlider::setBackgroundStyle( BackgroundStyles style )
{
    d_data->bgStyle = style;
    layoutSlider( true );
}

/*!
  \return the background style.
*/
QwtSlider::BackgroundStyles QwtSlider::backgroundStyle() const
{
    return d_data->bgStyle;
}

/*!
  \return QwtSlider::minimumSizeHint()
*/
QSize QwtSlider::sizeHint() const
{
    const QSize hint = minimumSizeHint();
    return hint.expandedTo( QApplication::globalStrut() );
}

/*!
  \brief Return a minimum size hint
  \warning The return value of QwtSlider::minimumSizeHint() depends on
           the font and the scale.
*/
QSize QwtSlider::minimumSizeHint() const
{
    if ( !d_data->sizeHintCache.isEmpty() )
        return d_data->sizeHintCache;

    const int minLength = 84; // from QSlider

    int handleLength = d_data->handleSize.width();
    int handleThickness = d_data->handleSize.height();

    if ( orientation() == Qt::Vertical )
        qSwap( handleLength, handleThickness );

    int w = minLength; 
    int h = handleThickness;

    if ( d_data->scalePos != QwtSlider::NoScale )
    {
        int d1, d2;
        scaleDraw()->getBorderDistHint( font(), d1, d2 );

        const int sdBorderDist = 2 * qMax( d1, d2 );
        const int sdExtent = qCeil( scaleDraw()->extent( font() ) );
        const int sdLength = scaleDraw()->minLength( font() );

        int l = sdLength;
        if ( handleLength > sdBorderDist )
        {
            // We need additional space for the overlapping handle
            l += handleLength - sdBorderDist;
        }

        w = qMax( l, w );
        h += sdExtent + d_data->spacing;
    }

    if ( d_data->bgStyle & QwtSlider::Trough )
        h += 2 * d_data->borderWidth;

    if ( orientation() == Qt::Vertical )
        qSwap( w, h );

    int left, right, top, bottom;
    getContentsMargins( &left, &top, &right, &bottom );

    w += left + right;
    h += top + bottom;

    d_data->sizeHintCache = QSize( w, h );
    return d_data->sizeHintCache;
}
