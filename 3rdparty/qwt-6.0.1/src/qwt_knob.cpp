/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_knob.h"
#include "qwt_round_scale_draw.h"
#include "qwt_math.h"
#include "qwt_painter.h"
#include <qpainter.h>
#include <qpalette.h>
#include <qstyle.h>
#include <qstyleoption.h>
#include <qevent.h>
#include <qmath.h>
#include <qapplication.h>

#if QT_VERSION < 0x040601
#define qAtan2(y, x) ::atan2(y, x)
#define qFabs(x) ::fabs(x)
#define qFastCos(x) ::cos(x)
#define qFastSin(x) ::sin(x)
#endif

class QwtKnob::PrivateData
{
public:
    PrivateData()
    {
        angle = 0.0;
        nTurns = 0.0;
        borderWidth = 2;
        borderDist = 4;
        totalAngle = 270.0;
        scaleDist = 4;
        markerStyle = QwtKnob::Notch;
        maxScaleTicks = 11;
        knobStyle = QwtKnob::Raised;
        knobWidth = 50;
        markerSize = 8;
    }

    QwtKnob::KnobStyle knobStyle;
    QwtKnob::MarkerStyle markerStyle;

    int borderWidth;
    int borderDist;
    int scaleDist;
    int maxScaleTicks;
    int knobWidth;
    int markerSize;

    double angle;
    double totalAngle;
    double nTurns;

    mutable QRectF knobRect; // bounding rect of the knob without scale
};

/*!
  Constructor
  \param parent Parent widget
*/
QwtKnob::QwtKnob( QWidget* parent ):
    QwtAbstractSlider( Qt::Horizontal, parent )
{
    initKnob();
}

void QwtKnob::initKnob()
{
    d_data = new PrivateData;

    setScaleDraw( new QwtRoundScaleDraw() );

    setUpdateTime( 50 );
    setTotalAngle( 270.0 );
    recalcAngle();
    setSizePolicy( QSizePolicy( QSizePolicy::Minimum, QSizePolicy::Minimum ) );

    setRange( 0.0, 10.0, 1.0 );
    setValue( 0.0 );
}

//! Destructor
QwtKnob::~QwtKnob()
{
    delete d_data;
}

/*!
  \brief Set the knob type 

  \param knobStyle Knob type
  \sa knobStyle(), setBorderWidth()
*/
void QwtKnob::setKnobStyle( KnobStyle knobStyle )
{
    if ( d_data->knobStyle != knobStyle )
    {
        d_data->knobStyle = knobStyle;
        update();
    }
}

/*!
    \return Marker type of the knob
    \sa setKnobStyle(), setBorderWidth()
*/
QwtKnob::KnobStyle QwtKnob::knobStyle() const
{
    return d_data->knobStyle;
}

/*!
  \brief Set the marker type of the knob

  \param markerStyle Marker type
  \sa markerStyle(), setMarkerSize()
*/
void QwtKnob::setMarkerStyle( MarkerStyle markerStyle )
{
    if ( d_data->markerStyle != markerStyle )
    {
        d_data->markerStyle = markerStyle;
        update();
    }
}

/*!
    \return Marker type of the knob
    \sa setMarkerStyle(), setMarkerSize()
*/
QwtKnob::MarkerStyle QwtKnob::markerStyle() const
{
    return d_data->markerStyle;
}

/*!
  \brief Set the total angle by which the knob can be turned
  \param angle Angle in degrees.

  The default angle is 270 degrees. It is possible to specify
  an angle of more than 360 degrees so that the knob can be
  turned several times around its axis.
*/
void QwtKnob::setTotalAngle ( double angle )
{
    if ( angle < 10.0 )
        d_data->totalAngle = 10.0;
    else
        d_data->totalAngle = angle;

    scaleDraw()->setAngleRange( -0.5 * d_data->totalAngle,
        0.5 * d_data->totalAngle );
    layoutKnob( true );
}

//! Return the total angle
double QwtKnob::totalAngle() const
{
    return d_data->totalAngle;
}

/*!
   Change the scale draw of the knob

   For changing the labels of the scales, it
   is necessary to derive from QwtRoundScaleDraw and
   overload QwtRoundScaleDraw::label().

   \sa scaleDraw()
*/
void QwtKnob::setScaleDraw( QwtRoundScaleDraw *scaleDraw )
{
    setAbstractScaleDraw( scaleDraw );
    setTotalAngle( d_data->totalAngle );
}

/*!
   \return the scale draw of the knob
   \sa setScaleDraw()
*/
const QwtRoundScaleDraw *QwtKnob::scaleDraw() const
{
    return static_cast<const QwtRoundScaleDraw *>( abstractScaleDraw() );
}

/*!
   \return the scale draw of the knob
   \sa setScaleDraw()
*/
QwtRoundScaleDraw *QwtKnob::scaleDraw()
{
    return static_cast<QwtRoundScaleDraw *>( abstractScaleDraw() );
}

/*!
  \brief Notify change of value

  Sets the knob's value to the nearest multiple
  of the step size.
*/
void QwtKnob::valueChange()
{
    recalcAngle();
    update();
    QwtAbstractSlider::valueChange();
}

/*!
  \brief Determine the value corresponding to a specified position

  Called by QwtAbstractSlider
  \param pos point
*/
double QwtKnob::getValue( const QPoint &pos )
{
    const double dx = rect().center().x() - pos.x();
    const double dy = rect().center().y() - pos.y();

    const double arc = qAtan2( -dx, dy ) * 180.0 / M_PI;

    double newValue =  0.5 * ( minValue() + maxValue() )
        + ( arc + d_data->nTurns * 360.0 ) * ( maxValue() - minValue() )
        / d_data->totalAngle;

    const double oneTurn = qFabs( maxValue() - minValue() ) * 360.0 / d_data->totalAngle;
    const double eqValue = value() + mouseOffset();

    if ( qFabs( newValue - eqValue ) > 0.5 * oneTurn )
    {
        if ( newValue < eqValue )
            newValue += oneTurn;
        else
            newValue -= oneTurn;
    }

    return newValue;
}

/*!
  \brief Set the scrolling mode and direction

  Called by QwtAbstractSlider
  \param pos Point in question
  \param scrollMode Scrolling mode
  \param direction Direction
*/
void QwtKnob::getScrollMode( const QPoint &pos, 
    QwtAbstractSlider::ScrollMode &scrollMode, int &direction ) const
{
    const int r = d_data->knobRect.width() / 2;

    const int dx = d_data->knobRect.x() + r - pos.x();
    const int dy = d_data->knobRect.y() + r - pos.y();

    if ( ( dx * dx ) + ( dy * dy ) <= ( r * r ) ) // point is inside the knob
    {
        scrollMode = QwtAbstractSlider::ScrMouse;
        direction = 0;
    }
    else                                // point lies outside
    {
        scrollMode = QwtAbstractSlider::ScrTimer;
        double arc = qAtan2( double( -dx ), double( dy ) ) * 180.0 / M_PI;
        if ( arc < d_data->angle )
            direction = -1;
        else if ( arc > d_data->angle )
            direction = 1;
        else
            direction = 0;
    }
}

/*!
  \brief Notify a change of the range

  Called by QwtAbstractSlider
*/
void QwtKnob::rangeChange()
{
    if ( autoScale() )
        rescale( minValue(), maxValue() );

    layoutKnob( true );
    recalcAngle();
}

/*!
  Qt Resize Event
  \param event Resize event
*/
void QwtKnob::resizeEvent( QResizeEvent *event )
{
    Q_UNUSED( event );
    layoutKnob( false );
}

/*! 
  Handle QEvent::StyleChange and QEvent::FontChange;
  \param event Change event
*/
void QwtKnob::changeEvent( QEvent *event )
{
    switch( event->type() )
    {
        case QEvent::StyleChange:
        case QEvent::FontChange:
            layoutKnob( true );
            break;
        default:
            break;
    }
}

/*!
   Recalculate the knob's geometry and layout based on
   the current rect and fonts.

   \param update_geometry notify the layout system and call update
                          to redraw the scale
*/
void QwtKnob::layoutKnob( bool update_geometry )
{
    const double d = d_data->knobWidth;

    d_data->knobRect.setWidth( d );
    d_data->knobRect.setHeight( d );
    d_data->knobRect.moveCenter( rect().center() );

    scaleDraw()->setRadius( 0.5 * d + d_data->scaleDist );
    scaleDraw()->moveCenter( rect().center() );

    if ( update_geometry )
    {
        updateGeometry();
        update();
    }
}

/*!
  Repaint the knob
  \param event Paint event
*/
void QwtKnob::paintEvent( QPaintEvent *event )
{
    QPainter painter( this );
    painter.setClipRegion( event->region() );

    QStyleOption opt;
    opt.init(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);

    painter.setRenderHint( QPainter::Antialiasing, true );

    if ( !d_data->knobRect.contains( event->region().boundingRect() ) )
        scaleDraw()->draw( &painter, palette() );

    drawKnob( &painter, d_data->knobRect );
    drawMarker( &painter, d_data->knobRect, d_data->angle );

    painter.setRenderHint( QPainter::Antialiasing, false );

    if ( hasFocus() )
        QwtPainter::drawFocusRect( &painter, this );
}

/*!
  \brief Draw the knob
  \param painter painter
  \param knobRect Bounding rectangle of the knob (without scale)
*/
void QwtKnob::drawKnob( QPainter *painter, 
    const QRectF &knobRect ) const
{
    double dim = qMin( knobRect.width(), knobRect.height() );
    dim -= d_data->borderWidth * 0.5;

    QRectF aRect( 0, 0, dim, dim );
    aRect.moveCenter( knobRect.center() );

    QPen pen( Qt::NoPen );
    if ( d_data->borderWidth > 0 )
    {
        QColor c1 = palette().color( QPalette::Light );
        QColor c2 = palette().color( QPalette::Dark );

        QLinearGradient gradient( aRect.topLeft(), aRect.bottomRight() );
        gradient.setColorAt( 0.0, c1 );
        gradient.setColorAt( 0.3, c1 );
        gradient.setColorAt( 0.7, c2 );
        gradient.setColorAt( 1.0, c2 );

        pen = QPen( gradient, d_data->borderWidth ); 
    }

    QBrush brush;
    switch( d_data->knobStyle )
    {
        case QwtKnob::Raised:
        {
            double off = 0.3 * knobRect.width();
            QRadialGradient gradient( knobRect.center(),
                knobRect.width(), knobRect.topLeft() + QPoint( off, off ) );
            
            gradient.setColorAt( 0.0, palette().color( QPalette::Midlight ) );
            gradient.setColorAt( 1.0, palette().color( QPalette::Button ) );

            brush = QBrush( gradient );

            break;
        }
        case QwtKnob::Sunken:
        {
            QLinearGradient gradient( 
                knobRect.topLeft(), knobRect.bottomRight() );
            gradient.setColorAt( 0.0, palette().color( QPalette::Mid ) );
            gradient.setColorAt( 0.5, palette().color( QPalette::Button ) );
            gradient.setColorAt( 1.0, palette().color( QPalette::Midlight ) );
            brush = QBrush( gradient );

            break;
        }
        default:
            brush = palette().brush( QPalette::Button );
    }

    painter->setPen( pen );
    painter->setBrush( brush );
    painter->drawEllipse( aRect );
}


/*!
  \brief Draw the marker at the knob's front
  \param painter Painter
  \param rect Bounding rectangle of the knob without scale
  \param angle Angle of the marker in degrees
*/
void QwtKnob::drawMarker( QPainter *painter, 
    const QRectF &rect, double angle ) const
{
    if ( d_data->markerStyle == NoMarker || !isValid() )
        return;

    const double radians = angle * M_PI / 180.0;
    const double sinA = -qFastSin( radians );
    const double cosA = qFastCos( radians );

    const double xm = rect.center().x();
    const double ym = rect.center().y();
    const double margin = 4.0;

    double radius = 0.5 * ( rect.width() - d_data->borderWidth ) - margin;
    if ( radius < 1.0 )
        radius = 1.0;

    switch ( d_data->markerStyle )
    {
        case Notch:
        case Nub:
        {
            const double dotWidth = 
                qMin( double( d_data->markerSize ), radius);

            const double dotCenterDist = radius - 0.5 * dotWidth;
            if ( dotCenterDist > 0.0 )
            {
                const QPointF center( xm - sinA * dotCenterDist, 
                    ym - cosA * dotCenterDist );

                QRectF ellipse( 0.0, 0.0, dotWidth, dotWidth );
                ellipse.moveCenter( center );

                QColor c1 = palette().color( QPalette::Light );
                QColor c2 = palette().color( QPalette::Mid );

                if ( d_data->markerStyle == Notch )
                    qSwap( c1, c2 );

                QLinearGradient gradient( 
                    ellipse.topLeft(), ellipse.bottomRight() );
                gradient.setColorAt( 0.0, c1 );
                gradient.setColorAt( 1.0, c2 );

                painter->setPen( Qt::NoPen );
                painter->setBrush( gradient );

                painter->drawEllipse( ellipse );
            }
            break;
        }
        case Dot:
        {
            const double dotWidth = 
                qMin( double( d_data->markerSize ), radius);

            const double dotCenterDist = radius - 0.5 * dotWidth;
            if ( dotCenterDist > 0.0 )
            {
                const QPointF center( xm - sinA * dotCenterDist, 
                    ym - cosA * dotCenterDist );

                QRectF ellipse( 0.0, 0.0, dotWidth, dotWidth );
                ellipse.moveCenter( center );

                painter->setPen( Qt::NoPen );
                painter->setBrush( palette().color( QPalette::ButtonText ) );
                painter->drawEllipse( ellipse );
            }

            break;
        }
        case Tick:
        {
            const double rb = qMax( radius - d_data->markerSize, 1.0 );
            const double re = radius;

            const QLine line( xm - sinA * rb, ym - cosA * rb,
                xm - sinA * re, ym - cosA * re );

            QPen pen( palette().color( QPalette::ButtonText ), 0 );
            pen.setCapStyle( Qt::FlatCap );
            painter->setPen( pen );
            painter->drawLine ( line );

            break;
        }
#if 0
        case Triangle:
        {
            const double rb = qMax( radius - d_data->markerSize, 1.0 );
            const double re = radius;

            painter->translate( rect.center() );
            painter->rotate( angle - 90.0 );
            
            QPolygonF polygon;
            polygon += QPointF( re, 0.0 );
            polygon += QPointF( rb, 0.5 * ( re - rb ) );
            polygon += QPointF( rb, -0.5 * ( re - rb ) );

            painter->setPen( Qt::NoPen );
            painter->setBrush( palette().color( QPalette::Text ) );
            painter->drawPolygon( polygon );
            break;
        }
#endif
        default:
            break;
    }
}

/*!
  \brief Change the knob's width.

  The specified width must be >= 5, or it will be clipped.
  \param width New width
*/
void QwtKnob::setKnobWidth( int width )
{
    d_data->knobWidth = qMax( width, 5 );
    layoutKnob( true );
}

//! Return the width of the knob
int QwtKnob::knobWidth() const
{
    return d_data->knobWidth;
}

/*!
  \brief Set the knob's border width
  \param borderWidth new border width
*/
void QwtKnob::setBorderWidth( int borderWidth )
{
    d_data->borderWidth = qMax( borderWidth, 0 );
    layoutKnob( true );
}

//! Return the border width
int QwtKnob::borderWidth() const
{
    return d_data->borderWidth;
}

/*!
  \brief Set the size of the marker
  \sa markerSize(), markerStyle()
*/
void QwtKnob::setMarkerSize( int size )
{
    if ( d_data->markerSize != size )
    {
        d_data->markerSize = size;
        update();
    }
}

//! Return the marker size
int QwtKnob::markerSize() const
{
    return d_data->markerSize;
}

/*!
  \brief Recalculate the marker angle corresponding to the
    current value
*/
void QwtKnob::recalcAngle()
{
    //
    // calculate the angle corresponding to the value
    //
    if ( maxValue() == minValue() )
    {
        d_data->angle = 0;
        d_data->nTurns = 0;
    }
    else
    {
        d_data->angle = ( value() - 0.5 * ( minValue() + maxValue() ) )
            / ( maxValue() - minValue() ) * d_data->totalAngle;
        d_data->nTurns = qFloor( ( d_data->angle + 180.0 ) / 360.0 );
        d_data->angle = d_data->angle - d_data->nTurns * 360.0;
    }
}


/*!
    Recalculates the layout
    \sa layoutKnob()
*/
void QwtKnob::scaleChange()
{
    layoutKnob( true );
}

/*!
  \return minimumSizeHint()
*/
QSize QwtKnob::sizeHint() const
{
    const QSize hint = minimumSizeHint();
    return hint.expandedTo( QApplication::globalStrut() );
}

/*!
  \brief Return a minimum size hint
  \warning The return value of QwtKnob::minimumSizeHint() depends on the
           font and the scale.
*/
QSize QwtKnob::minimumSizeHint() const
{
    // Add the scale radial thickness to the knobWidth
    const int sh = qCeil( scaleDraw()->extent( font() ) );
    const int d = 2 * sh + 2 * d_data->scaleDist + d_data->knobWidth;

    return QSize( d, d );
}
