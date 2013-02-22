/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_thermo.h"
#include "qwt_scale_engine.h"
#include "qwt_scale_draw.h"
#include "qwt_scale_map.h"
#include "qwt_color_map.h"
#include <qpainter.h>
#include <qevent.h>
#include <qdrawutil.h>
#include <qstyle.h>
#include <qstyleoption.h>
#include <qmath.h>

static inline void qwtDrawLine( 
    QPainter *painter, int pos, 
    const QColor &color, const QRect pipeRect, 
    Qt::Orientation orientation )
{
    painter->setPen( color );
    if ( orientation == Qt::Horizontal )
        painter->drawLine( pos, pipeRect.top(), pos, pipeRect.bottom() );
    else
        painter->drawLine( pipeRect.left(), pos, pipeRect.right(), pos );
}

QVector<double> qwtTickList( const QwtScaleDiv &scaleDiv, double value )
{
    QVector<double> values;

    double lowerLimit = scaleDiv.interval().minValue();
    double upperLimit = scaleDiv.interval().maxValue();

    if ( upperLimit < lowerLimit )
        qSwap( lowerLimit, upperLimit );

    if ( value < lowerLimit )
        return values;

    if ( value < upperLimit )
        upperLimit = value;

    values += lowerLimit;

    for ( int tickType = QwtScaleDiv::MinorTick;
        tickType < QwtScaleDiv::NTickTypes; tickType++ )
    {
        const QList<double> ticks = scaleDiv.ticks( tickType );

        for ( int i = 0; i < ticks.count(); i++ )
        {
            const double v = ticks[i];
            if ( v > lowerLimit && v < upperLimit )
                values += v;
        }       
    }   

    values += upperLimit;
    
    return values;
}

class QwtThermo::PrivateData
{
public:
    PrivateData():
        orientation( Qt::Vertical ),
        scalePos( QwtThermo::LeftScale ),
        spacing( 3 ),
        borderWidth( 2 ),
        pipeWidth( 10 ),
        alarmLevel( 0.0 ),
        alarmEnabled( false ),
        autoFillPipe( true ),
        colorMap( NULL ),
        value( 0.0 )
    {
        rangeFlags = QwtInterval::IncludeBorders;
    }

    ~PrivateData()
    {
        delete colorMap;
    }

    Qt::Orientation orientation;
    ScalePos scalePos;
    int spacing;
    int borderWidth;
    int pipeWidth;

    QwtInterval::BorderFlags rangeFlags;
    double alarmLevel;
    bool alarmEnabled;
    bool autoFillPipe;

    QwtColorMap *colorMap;

    double value;
};

/*!
  Constructor
  \param parent Parent widget
*/
QwtThermo::QwtThermo( QWidget *parent ):
    QwtAbstractScale( parent )
{
    d_data = new PrivateData;

    QSizePolicy policy( QSizePolicy::MinimumExpanding, QSizePolicy::Fixed );
    if ( d_data->orientation == Qt::Vertical )
        policy.transpose();

    setSizePolicy( policy );

    setAttribute( Qt::WA_WState_OwnSizePolicy, false );
    layoutThermo( true );
}

//! Destructor
QwtThermo::~QwtThermo()
{
    delete d_data;
}

/*!
  \brief Exclude/Include min/max values

  According to the flags minValue() and maxValue()
  are included/excluded from the pipe. In case of an
  excluded value the corresponding tick is painted
  1 pixel off of the pipeRect().

  F.e. when a minimum
  of 0.0 has to be displayed as an empty pipe the minValue()
  needs to be excluded.

  \param flags Range flags
  \sa rangeFlags()
*/
void QwtThermo::setRangeFlags( QwtInterval::BorderFlags flags )
{
    if ( d_data->rangeFlags != flags )
    {
        d_data->rangeFlags = flags;
        update();
    }
}

/*!
  \return Range flags
  \sa setRangeFlags()
*/
QwtInterval::BorderFlags QwtThermo::rangeFlags() const
{
    return d_data->rangeFlags;
}

/*!
  Set the current value.

  \param value New Value
  \sa value()
*/
void QwtThermo::setValue( double value )
{
    if ( d_data->value != value )
    {
        d_data->value = value;
        update();
    }
}

//! Return the value.
double QwtThermo::value() const
{
    return d_data->value;
}

/*!
  \brief Set a scale draw

  For changing the labels of the scales, it
  is necessary to derive from QwtScaleDraw and
  overload QwtScaleDraw::label().

  \param scaleDraw ScaleDraw object, that has to be created with
                   new and will be deleted in ~QwtThermo() or the next
                   call of setScaleDraw().
*/
void QwtThermo::setScaleDraw( QwtScaleDraw *scaleDraw )
{
    setAbstractScaleDraw( scaleDraw );
}

/*!
   \return the scale draw of the thermo
   \sa setScaleDraw()
*/
const QwtScaleDraw *QwtThermo::scaleDraw() const
{
    return static_cast<const QwtScaleDraw *>( abstractScaleDraw() );
}

/*!
   \return the scale draw of the thermo
   \sa setScaleDraw()
*/
QwtScaleDraw *QwtThermo::scaleDraw()
{
    return static_cast<QwtScaleDraw *>( abstractScaleDraw() );
}

/*!
  Paint event handler
  \param event Paint event
*/
void QwtThermo::paintEvent( QPaintEvent *event )
{
    QPainter painter( this );
    painter.setClipRegion( event->region() );

    QStyleOption opt;
    opt.init(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);

    const QRect tRect = pipeRect();

    if ( !tRect.contains( event->rect() ) )
    {
        if ( d_data->scalePos != NoScale )
            scaleDraw()->draw( &painter, palette() );
    }

    const int bw = d_data->borderWidth;

    const QBrush brush = palette().brush( QPalette::Base );
    qDrawShadePanel( &painter, 
        tRect.adjusted( -bw, -bw, bw, bw ),
        palette(), true, bw, 
        d_data->autoFillPipe ? &brush : NULL );

    drawLiquid( &painter, tRect );
}

/*! 
  Resize event handler
  \param event Resize event
*/
void QwtThermo::resizeEvent( QResizeEvent *event )
{
    Q_UNUSED( event );
    layoutThermo( false );
}

/*! 
  Qt change event handler
  \param event Event
*/
void QwtThermo::changeEvent( QEvent *event )
{
    switch( event->type() )
    {
        case QEvent::StyleChange:
        case QEvent::FontChange:
        {
            layoutThermo( true );
            break;
        }
        default:
            break;
    }
}

/*!
  Recalculate the QwtThermo geometry and layout based on
  the QwtThermo::contentsRect() and the fonts.

  \param update_geometry notify the layout system and call update
         to redraw the scale
*/
void QwtThermo::layoutThermo( bool update_geometry )
{
    const QRect tRect = pipeRect();
    const int bw = d_data->borderWidth + d_data->spacing;
    const bool inverted = ( upperBound() < lowerBound() );

    int from, to;

    if ( d_data->orientation == Qt::Horizontal )
    {
        from = tRect.left();
        to = tRect.right();

        if ( d_data->rangeFlags & QwtInterval::ExcludeMinimum )
        {
            if ( inverted )
                to++;
            else
                from--;
        }
        if ( d_data->rangeFlags & QwtInterval::ExcludeMaximum )
        {
            if ( inverted )
                from--;
            else
                to++;
        }

        switch ( d_data->scalePos )
        {
            case TopScale:
            {
                scaleDraw()->setAlignment( QwtScaleDraw::TopScale );
                scaleDraw()->move( from, tRect.top() - bw );
                scaleDraw()->setLength( to - from );
                break;
            }

            case BottomScale:
            case NoScale: 
            default:
            {
                scaleDraw()->setAlignment( QwtScaleDraw::BottomScale );
                scaleDraw()->move( from, tRect.bottom() + bw );
                scaleDraw()->setLength( to - from );
                break;
            }
        }
    }
    else // Qt::Vertical
    {
        from = tRect.top();
        to = tRect.bottom();

        if ( d_data->rangeFlags & QwtInterval::ExcludeMinimum )
        {
            if ( inverted )
                from--;
            else
                to++;
        }
        if ( d_data->rangeFlags & QwtInterval::ExcludeMaximum )
        {
            if ( inverted )
                to++;
            else
                from--;
        }

        switch ( d_data->scalePos )
        {
            case RightScale:
            {
                scaleDraw()->setAlignment( QwtScaleDraw::RightScale );
                scaleDraw()->move( tRect.right() + bw, from );
                scaleDraw()->setLength( to - from );
                break;
            }

            case LeftScale:
            case NoScale: 
            default:
            {
                scaleDraw()->setAlignment( QwtScaleDraw::LeftScale );
                scaleDraw()->move( tRect.left() - bw, from );
                scaleDraw()->setLength( to - from );
                break;
            }
        }
    }

    if ( update_geometry )
    {
        updateGeometry();
        update();
    }
}

/*!
  \return Bounding rectangle of the pipe ( without borders )
          in widget coordinates
*/
QRect QwtThermo::pipeRect() const
{
    const QRect cr = contentsRect();

    int mbd = 0;
    if ( d_data->scalePos != NoScale )
    {
        int d1, d2;
        scaleDraw()->getBorderDistHint( font(), d1, d2 );
        mbd = qMax( d1, d2 );
    }
    const int bw = d_data->borderWidth;

    QRect tRect;
    if ( d_data->orientation == Qt::Horizontal )
    {
        switch ( d_data->scalePos )
        {
            case TopScale:
            {
                tRect.setRect(
                    cr.x() + mbd + bw,
                    cr.y() + cr.height() - d_data->pipeWidth - bw,
                    cr.width() - 2 * ( bw + mbd ),
                    d_data->pipeWidth 
                );
                break;
            }

            case BottomScale:
            case NoScale: 
            default:   
            {
                tRect.setRect(
                    cr.x() + mbd + bw,
                    cr.y() + bw,
                    cr.width() - 2 * ( bw + mbd ),
                    d_data->pipeWidth 
                );
                break;
            }
        }
    }
    else // Qt::Vertical
    {
        switch ( d_data->scalePos )
        {
            case RightScale:
            {
                tRect.setRect(
                    cr.x() + bw,
                    cr.y() + mbd + bw,
                    d_data->pipeWidth,
                    cr.height() - 2 * ( bw + mbd ) 
                );
                break;
            }
            case LeftScale:
            case NoScale: 
            default:   
            {
                tRect.setRect(
                    cr.x() + cr.width() - bw - d_data->pipeWidth,
                    cr.y() + mbd + bw,
                    d_data->pipeWidth,
                    cr.height() - 2 * ( bw + mbd ) );
                break;
            }
        }
    }

    return tRect;
}

/*!
   \brief Set the thermometer orientation and the scale position.

   The scale position NoScale disables the scale.
   \param o orientation. Possible values are Qt::Horizontal and Qt::Vertical.
         The default value is Qt::Vertical.
   \param s Position of the scale.
         The default value is NoScale.

   A valid combination of scale position and orientation is enforced:
   - a horizontal thermometer can have the scale positions TopScale,
     BottomScale or NoScale;
   - a vertical thermometer can have the scale positions LeftScale,
     RightScale or NoScale;
   - an invalid scale position will default to NoScale.

   \sa setScalePosition()
*/
void QwtThermo::setOrientation( Qt::Orientation o, ScalePos s )
{
    if ( o == d_data->orientation && s == d_data->scalePos )
        return;

    switch ( o )
    {
        case Qt::Horizontal:
        {
            if ( ( s == NoScale ) || ( s == BottomScale ) || ( s == TopScale ) )
                d_data->scalePos = s;
            else
                d_data->scalePos = NoScale;
            break;
        }
        case Qt::Vertical:
        {
            if ( ( s == NoScale ) || ( s == LeftScale ) || ( s == RightScale ) )
                d_data->scalePos = s;
            else
                d_data->scalePos = NoScale;
            break;
        }
    }

    if ( o != d_data->orientation )
    {
        if ( !testAttribute( Qt::WA_WState_OwnSizePolicy ) )
        {
            QSizePolicy sp = sizePolicy();
            sp.transpose();
            setSizePolicy( sp );

            setAttribute( Qt::WA_WState_OwnSizePolicy, false );
        }
    }

    d_data->orientation = o;
    layoutThermo( true );
}

/*!
  \brief Change the scale position (and thermometer orientation).

  \param scalePos Position of the scale.

  A valid combination of scale position and orientation is enforced:
  - if the new scale position is LeftScale or RightScale, the
    scale orientation will become Qt::Vertical;
  - if the new scale position is BottomScale or TopScale, the scale
    orientation will become Qt::Horizontal;
  - if the new scale position is NoScale, the scale orientation 
    will not change.

  \sa setOrientation(), scalePosition()
*/
void QwtThermo::setScalePosition( ScalePos scalePos )
{
    if ( ( scalePos == BottomScale ) || ( scalePos == TopScale ) )
        setOrientation( Qt::Horizontal, scalePos );
    else if ( ( scalePos == LeftScale ) || ( scalePos == RightScale ) )
        setOrientation( Qt::Vertical, scalePos );
    else
        setOrientation( d_data->orientation, NoScale );
}

/*!
   Return the scale position.
   \sa setScalePosition()
*/
QwtThermo::ScalePos QwtThermo::scalePosition() const
{
    return d_data->scalePos;
}

//! Notify a scale change.
void QwtThermo::scaleChange()
{
    layoutThermo( true );
}

/*!
   Redraw the liquid in thermometer pipe.
   \param painter Painter
   \param pipeRect Bounding rectangle of the pipe without borders
*/
void QwtThermo::drawLiquid( 
    QPainter *painter, const QRect &pipeRect ) const
{
    painter->save();
    painter->setClipRect( pipeRect, Qt::IntersectClip );

    const bool inverted = ( upperBound() < lowerBound() );

    const QwtScaleMap scaleMap = scaleDraw()->scaleMap();

    if ( d_data->colorMap != NULL )
    {
        const QwtInterval interval = scaleDiv().interval().normalized();

        // Because the positions of the ticks are rounded
        // we calculate the colors for the rounded tick values

        QVector<double> values = qwtTickList(
            scaleDraw()->scaleDiv(), d_data->value );

        if ( scaleMap.isInverting() )
            qSort( values.begin(), values.end(), qGreater<double>() );
        else
            qSort( values.begin(), values.end(), qLess<double>() );

        int from;
        if ( !values.isEmpty() )
        {
            from = qRound( scaleMap.transform( values[0] ) );
            qwtDrawLine( painter, from,
                d_data->colorMap->color( interval, values[0] ),
                pipeRect, d_data->orientation );
        }

        for ( int i = 1; i < values.size(); i++ )
        {
            const int to = qRound( scaleMap.transform( values[i] ) );

            for ( int pos = from + 1; pos < to; pos++ )
            {
                const double v = scaleMap.invTransform( pos );

                qwtDrawLine( painter, pos, 
                    d_data->colorMap->color( interval, v ),
                    pipeRect, d_data->orientation );
            }
            qwtDrawLine( painter, to,
                d_data->colorMap->color( interval, values[i] ),
                pipeRect, d_data->orientation );

            from = to;
        }
    }
    else
    {
        const int tval = qRound( scaleMap.transform( d_data->value ) );

        QRect fillRect = pipeRect;
        if ( d_data->orientation == Qt::Horizontal )
        {
            if ( inverted )
                fillRect.setLeft( tval );
            else
                fillRect.setRight( tval );
        }
        else // Qt::Vertical
        {
            if ( inverted )
                fillRect.setBottom( tval );
            else
                fillRect.setTop( tval );
        }

        if ( d_data->alarmEnabled &&
            d_data->value >= d_data->alarmLevel )
        {
            QRect alarmRect = fillRect;

            const int taval = qRound( scaleMap.transform( d_data->alarmLevel ) );
            if ( d_data->orientation == Qt::Horizontal )
            {
                if ( inverted )
                    alarmRect.setRight( taval );
                else
                    alarmRect.setLeft( taval );
            }
            else
            {
                if ( inverted )
                    alarmRect.setTop( taval );
                else
                    alarmRect.setBottom( taval );
            }

            fillRect = QRegion( fillRect ).subtracted( alarmRect ).boundingRect();

            painter->fillRect( alarmRect, palette().brush( QPalette::Highlight ) );
        }

        painter->fillRect( fillRect, palette().brush( QPalette::ButtonText ) );
    }

    painter->restore();
}

/*!
  \brief Change the spacing between pipe and scale

  A spacing of 0 means, that the backbone of the scale is below
  the pipe.

  The default setting is 3 pixels.

  \param spacing Number of pixels
  \sa spacing();
*/
void QwtThermo::setSpacing( int spacing )
{
    if ( spacing <= 0 )
        spacing = 0;

    if ( spacing != d_data->spacing  )
    {
        d_data->spacing = spacing;
        layoutThermo( true );
    }
}

/*!
  \return Number of pixels between pipe and scale
  \sa setSpacing()
*/
int QwtThermo::spacing() const
{
    return d_data->spacing;
}

/*!
   Set the border width of the pipe.
   \param width Border width
   \sa borderWidth()
*/
void QwtThermo::setBorderWidth( int width )
{
    if ( width <= 0 )
        width = 0;

    if ( width != d_data->borderWidth  )
    {
        d_data->borderWidth = width;
        layoutThermo( true );
    }
}

/*!
   Return the border width of the thermometer pipe.
   \sa setBorderWidth()
*/
int QwtThermo::borderWidth() const
{
    return d_data->borderWidth;
}

/*!
  \brief Assign a color map for the fill color

  \param colorMap Color map
  \warning The alarm threshold has no effect, when
           a color map has been assigned
*/
void QwtThermo::setColorMap( QwtColorMap *colorMap )
{
    if ( colorMap != d_data->colorMap )
    {
        delete d_data->colorMap;
        d_data->colorMap = colorMap;
    }
}

/*!
  \return Color map for the fill color
  \warning The alarm threshold has no effect, when
           a color map has been assigned
*/
QwtColorMap *QwtThermo::colorMap()
{
    return d_data->colorMap;
}

/*!
  \return Color map for the fill color
  \warning The alarm threshold has no effect, when
           a color map has been assigned
*/
const QwtColorMap *QwtThermo::colorMap() const
{
    return d_data->colorMap;
}

/*!
  \brief Change the brush of the liquid.
 
  Changes the QPalette::ButtonText brush of the palette.

  \param brush New brush. 
  \sa fillBrush(), QWidget::setPalette()
*/
void QwtThermo::setFillBrush( const QBrush& brush )
{
    QPalette pal = palette();
    pal.setBrush( QPalette::ButtonText, brush );
    setPalette( pal );
}

/*!
  Return the liquid ( QPalette::ButtonText ) brush. 
  \sa setFillBrush(), QWidget::palette()
*/
const QBrush& QwtThermo::fillBrush() const
{
    return palette().brush( QPalette::ButtonText );
}

/*!
  \brief Specify the liquid brush above the alarm threshold

  Changes the QPalette::Highlight brush of the palette.

  \param brush New brush. 
  \sa alarmBrush(), QWidget::setPalette()

  \warning The alarm threshold has no effect, when
           a color map has been assigned
*/
void QwtThermo::setAlarmBrush( const QBrush& brush )
{
    QPalette pal = palette();
    pal.setBrush( QPalette::Highlight, brush );
    setPalette( pal );
}

/*!
  Return the liquid brush ( QPalette::Highlight ) above the alarm threshold.
  \sa setAlarmBrush(), QWidget::palette()

  \warning The alarm threshold has no effect, when
           a color map has been assigned
*/
const QBrush& QwtThermo::alarmBrush() const
{
    return palette().brush( QPalette::Highlight );
}

/*!
  Specify the alarm threshold.

  \param level Alarm threshold
  \sa alarmLevel()

  \warning The alarm threshold has no effect, when
           a color map has been assigned
*/
void QwtThermo::setAlarmLevel( double level )
{
    d_data->alarmLevel = level;
    d_data->alarmEnabled = 1;
    update();
}

/*!
  Return the alarm threshold.
  \sa setAlarmLevel()

  \warning The alarm threshold has no effect, when
           a color map has been assigned
*/
double QwtThermo::alarmLevel() const
{
    return d_data->alarmLevel;
}

/*!
  Change the width of the pipe.

  \param width Width of the pipe
  \sa pipeWidth()
*/
void QwtThermo::setPipeWidth( int width )
{
    if ( width > 0 )
    {
        d_data->pipeWidth = width;
        layoutThermo( true );
    }
}

/*!
  Return the width of the pipe.
  \sa setPipeWidth()
*/
int QwtThermo::pipeWidth() const
{
    return d_data->pipeWidth;
}

/*!
  \brief Enable or disable the alarm threshold
  \param on true (disabled) or false (enabled)

  \warning The alarm threshold has no effect, when
           a color map has been assigned
*/
void QwtThermo::setAlarmEnabled( bool on )
{
    d_data->alarmEnabled = on;
    update();
}

/*! 
  \return True, when the alarm threshold is enabled.

  \warning The alarm threshold has no effect, when
           a color map has been assigned
*/
bool QwtThermo::alarmEnabled() const
{
    return d_data->alarmEnabled;
}

/*!
  \return the minimum size hint
  \sa minimumSizeHint()
*/
QSize QwtThermo::sizeHint() const
{
    return minimumSizeHint();
}

/*!
  \brief Return a minimum size hint
  \warning The return value depends on the font and the scale.
  \sa sizeHint()
*/
QSize QwtThermo::minimumSizeHint() const
{
    int w = 0, h = 0;

    if ( d_data->scalePos != NoScale )
    {
        const int sdExtent = qCeil( scaleDraw()->extent( font() ) );
        const int sdLength = scaleDraw()->minLength( font() );

        w = sdLength;
        h = d_data->pipeWidth + sdExtent + d_data->spacing;

    }
    else // no scale
    {
        w = 200;
        h = d_data->pipeWidth;
    }

    if ( d_data->orientation == Qt::Vertical )
        qSwap( w, h );

    w += 2 * d_data->borderWidth;
    h += 2 * d_data->borderWidth;

    // finally add the margins
    int left, right, top, bottom;
    getContentsMargins( &left, &top, &right, &bottom );
    w += left + right;
    h += top + bottom;

    return QSize( w, h );
}
