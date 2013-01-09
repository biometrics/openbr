/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_wheel.h"
#include "qwt_math.h"
#include "qwt_painter.h"
#include <qevent.h>
#include <qdrawutil.h>
#include <qpainter.h>
#include <qstyle.h>
#include <qstyleoption.h>
#include <qapplication.h>

#if QT_VERSION < 0x040601
#define qFastSin(x) ::sin(x)
#endif

class QwtWheel::PrivateData
{
public:
    PrivateData()
    {
        viewAngle = 175.0;
        totalAngle = 360.0;
        tickCnt = 10;
        wheelBorderWidth = 2;
        borderWidth = 2;
        wheelWidth = 20;
    };

    double viewAngle;
    double totalAngle;
    int tickCnt;
    int wheelBorderWidth;
    int borderWidth;
    int wheelWidth;
};

//! Constructor
QwtWheel::QwtWheel( QWidget *parent ):
    QwtAbstractSlider( Qt::Horizontal, parent )
{
    d_data = new PrivateData;

    setSizePolicy( QSizePolicy::Preferred, QSizePolicy::Fixed );

    setAttribute( Qt::WA_WState_OwnSizePolicy, false );
    setUpdateTime( 50 );
}

//! Destructor
QwtWheel::~QwtWheel()
{
    delete d_data;
}

/*!
  \brief Adjust the number of grooves in the wheel's surface.

  The number of grooves is limited to 6 <= cnt <= 50.
  Values outside this range will be clipped.
  The default value is 10.

  \param cnt Number of grooves per 360 degrees
  \sa tickCnt()
*/
void QwtWheel::setTickCnt( int cnt )
{
    d_data->tickCnt = qBound( 6, cnt, 50 );
    update();
}

/*!
  \return Number of grooves in the wheel's surface.
  \sa setTickCnt()
*/
int QwtWheel::tickCnt() const
{
    return d_data->tickCnt;
}

/*!
    \return mass
*/
double QwtWheel::mass() const
{
    return QwtAbstractSlider::mass();
}

/*!
  \brief Set the wheel border width of the wheel.

  The wheel border must not be smaller than 1
  and is limited in dependence on the wheel's size.
  Values outside the allowed range will be clipped.

  The wheel border defaults to 2.

  \param borderWidth Border width
  \sa internalBorder()
*/
void QwtWheel::setWheelBorderWidth( int borderWidth )
{
    const int d = qMin( width(), height() ) / 3;
    borderWidth = qMin( borderWidth, d );
    d_data->wheelBorderWidth = qMax( borderWidth, 1 );
    update();
}

/*!
   \return Wheel border width 
   \sa setWheelBorderWidth()
*/
int QwtWheel::wheelBorderWidth() const
{
    return d_data->wheelBorderWidth;
}

/*!
  \brief Set the border width 

  The border defaults to 2.

  \param width Border width
  \sa borderWidth()
*/
void QwtWheel::setBorderWidth( int width )
{
    d_data->borderWidth = qMax( width, 0 );
    update();
}

/*!
   \return Border width 
   \sa setBorderWidth()
*/
int QwtWheel::borderWidth() const
{
    return d_data->borderWidth;
}

/*!
   \return Rectangle of the wheel without the outer border
*/
QRect QwtWheel::wheelRect() const
{
    const int bw = d_data->borderWidth;
    return contentsRect().adjusted( bw, bw, -bw, -bw );
}

/*!
  \brief Set the total angle which the wheel can be turned.

  One full turn of the wheel corresponds to an angle of
  360 degrees. A total angle of n*360 degrees means
  that the wheel has to be turned n times around its axis
  to get from the minimum value to the maximum value.

  The default setting of the total angle is 360 degrees.

  \param angle total angle in degrees
  \sa totalAngle()
*/
void QwtWheel::setTotalAngle( double angle )
{
    if ( angle < 0.0 )
        angle = 0.0;

    d_data->totalAngle = angle;
    update();
}

/*!
  \return Total angle which the wheel can be turned.
  \sa setTotalAngle()
*/
double QwtWheel::totalAngle() const
{
    return d_data->totalAngle;
}

/*!
  \brief Set the wheel's orientation.
  \param o Orientation. Allowed values are
           Qt::Horizontal and Qt::Vertical.
   Defaults to Qt::Horizontal.
  \sa QwtAbstractSlider::orientation()
*/
void QwtWheel::setOrientation( Qt::Orientation o )
{
    if ( orientation() == o )
        return;

    if ( !testAttribute( Qt::WA_WState_OwnSizePolicy ) )
    {
        QSizePolicy sp = sizePolicy();
        sp.transpose();
        setSizePolicy( sp );

        setAttribute( Qt::WA_WState_OwnSizePolicy, false );
    }

    QwtAbstractSlider::setOrientation( o );
    update();
}

/*!
  \brief Specify the visible portion of the wheel.

  You may use this function for fine-tuning the appearance of
  the wheel. The default value is 175 degrees. The value is
  limited from 10 to 175 degrees.

  \param angle Visible angle in degrees
  \sa viewAngle(), setTotalAngle()
*/
void QwtWheel::setViewAngle( double angle )
{
    d_data->viewAngle = qBound( 10.0, angle, 175.0 );
    update();
}

/*!
  \return Visible portion of the wheel
  \sa setViewAngle(), totalAngle()
*/
double QwtWheel::viewAngle() const
{
    return d_data->viewAngle;
}

//! Determine the value corresponding to a specified point
double QwtWheel::getValue( const QPoint &p )
{
    const QRectF rect = wheelRect();

    // The reference position is arbitrary, but the
    // sign of the offset is important
    double w, dx;
    if ( orientation() == Qt::Vertical )
    {
        w = rect.height();
        dx = rect.y() - p.y();
    }
    else
    {
        w = rect.width();
        dx = p.x() - rect.x();
    }

    if ( w == 0.0 )
        return 0.0;

    // w pixels is an arc of viewAngle degrees,
    // so we convert change in pixels to change in angle
    const double ang = dx * d_data->viewAngle / w;

    // value range maps to totalAngle degrees,
    // so convert the change in angle to a change in value
    const double val = ang * ( maxValue() - minValue() ) / d_data->totalAngle;

    // Note, range clamping and rasterizing to step is automatically
    // handled by QwtAbstractSlider, so we simply return the change in value
    return val;
}

/*! 
   \brief Qt Resize Event
   \param event Resize event
*/
void QwtWheel::resizeEvent( QResizeEvent *event )
{
    QwtAbstractSlider::resizeEvent( event );
}

/*! 
   \brief Qt Paint Event
   \param event Paint event
*/
void QwtWheel::paintEvent( QPaintEvent *event )
{
    QPainter painter( this );
    painter.setClipRegion( event->region() );

    QStyleOption opt;
    opt.init(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);

    qDrawShadePanel( &painter, 
        contentsRect(), palette(), true, d_data->borderWidth );

    drawWheelBackground( &painter, wheelRect() );
    drawTicks( &painter, wheelRect() );

    if ( hasFocus() )
        QwtPainter::drawFocusRect( &painter, this );
}

/*!
   Draw the Wheel's background gradient

   \param painter Painter
   \param rect Rectangle for the wheel
*/
void QwtWheel::drawWheelBackground( 
    QPainter *painter, const QRectF &rect )
{
    painter->save();

    QPalette pal = palette();

    //  draw shaded background
    QLinearGradient gradient( rect.topLeft(), 
        ( orientation() == Qt::Horizontal ) ? rect.topRight() : rect.bottomLeft() );
    gradient.setColorAt( 0.0, pal.color( QPalette::Button ) );
    gradient.setColorAt( 0.2, pal.color( QPalette::Light ) );
    gradient.setColorAt( 0.7, pal.color( QPalette::Mid ) );
    gradient.setColorAt( 1.0, pal.color( QPalette::Dark ) );

    painter->fillRect( rect, gradient );

    // draw internal border

    const QPen lightPen( palette().color( QPalette::Light ), 
        d_data->wheelBorderWidth, Qt::SolidLine, Qt::FlatCap );
    const QPen darkPen( pal.color( QPalette::Dark ), 
        d_data->wheelBorderWidth, Qt::SolidLine, Qt::FlatCap );

    const double bw2 = 0.5 * d_data->wheelBorderWidth;

    if ( orientation() == Qt::Horizontal )
    {
        painter->setPen( lightPen );
        painter->drawLine( rect.left(), rect.top() + bw2, 
            rect.right(), rect.top() + bw2 );

        painter->setPen( darkPen );
        painter->drawLine( rect.left(), rect.bottom() - bw2, 
            rect.right(), rect.bottom() - bw2 );
    }
    else // Qt::Vertical
    {
        painter->setPen( lightPen );
        painter->drawLine( rect.left() + bw2, rect.top(), 
            rect.left() + bw2, rect.bottom() );

        painter->setPen( darkPen );
        painter->drawLine( rect.right() - bw2, rect.top(), 
            rect.right() - bw2, rect.bottom() );
    }

    painter->restore();
}

/*!
   Draw the Wheel's ticks

   \param painter Painter
   \param rect Rectangle for the wheel
*/
void QwtWheel::drawTicks( QPainter *painter, const QRectF &rect )
{
    if ( maxValue() == minValue() || d_data->totalAngle == 0.0 )
        return;

    const QPen lightPen( palette().color( QPalette::Light ), 
        0, Qt::SolidLine, Qt::FlatCap );
    const QPen darkPen( palette().color( QPalette::Dark ), 
        0, Qt::SolidLine, Qt::FlatCap );

    const double sign = ( minValue() < maxValue() ) ? 1.0 : -1.0;
    const double cnvFactor = qAbs( d_data->totalAngle / ( maxValue() - minValue() ) );
    const double halfIntv = 0.5 * d_data->viewAngle / cnvFactor;
    const double loValue = value() - halfIntv;
    const double hiValue = value() + halfIntv;
    const double tickWidth = 360.0 / double( d_data->tickCnt ) / cnvFactor;
    const double sinArc = qFastSin( d_data->viewAngle * M_PI / 360.0 );

    if ( orientation() == Qt::Horizontal )
    {
        const double halfSize = rect.width() * 0.5;

        double l1 = rect.top() + d_data->wheelBorderWidth;
        double l2 = rect.bottom() - d_data->wheelBorderWidth - 1;

        // draw one point over the border if border > 1
        if ( d_data->wheelBorderWidth > 1 )
        {
            l1--;
            l2++;
        }

        const double maxpos = rect.right() - 2;
        const double minpos = rect.left() + 2;

        // draw tick marks
        for ( double tickValue = qwtCeilF( loValue / tickWidth ) * tickWidth;
            tickValue < hiValue; tickValue += tickWidth )
        {
            const double angle = ( tickValue - value() ) * M_PI / 180.0;
            const double s = qFastSin( angle * cnvFactor );

            const double tickPos = 
                rect.right() - halfSize * ( sinArc + sign * s ) / sinArc;

            if ( ( tickPos <= maxpos ) && ( tickPos > minpos ) )
            {
                painter->setPen( darkPen );
                painter->drawLine( tickPos - 1 , l1, tickPos - 1,  l2 );
                painter->setPen( lightPen );
                painter->drawLine( tickPos, l1, tickPos, l2 );
            }
        }
    }
    else // Qt::Vertical
    {
        const double halfSize = rect.height() * 0.5;

        double l1 = rect.left() + d_data->wheelBorderWidth;
        double l2 = rect.right() - d_data->wheelBorderWidth - 1;

        if ( d_data->wheelBorderWidth > 1 )
        {
            l1--;
            l2++;
        }

        const double maxpos = rect.bottom() - 2;
        const double minpos = rect.top() + 2;

        for ( double tickValue = qwtCeilF( loValue / tickWidth ) * tickWidth;
            tickValue < hiValue; tickValue += tickWidth )
        {
            const double angle = ( tickValue - value() ) * M_PI / 180.0;
            const double s = qFastSin( angle * cnvFactor );

            const double tickPos = 
                rect.y() + halfSize * ( sinArc + sign * s ) / sinArc;

            if ( ( tickPos <= maxpos ) && ( tickPos > minpos ) )
            {
                painter->setPen( darkPen );
                painter->drawLine( l1, tickPos - 1 , l2, tickPos - 1 );
                painter->setPen( lightPen );
                painter->drawLine( l1, tickPos, l2, tickPos );
            }
        }
    }
}

//! Notify value change
void QwtWheel::valueChange()
{
    QwtAbstractSlider::valueChange();
    update();
}

/*!
  \brief Determine the scrolling mode and direction corresponding
         to a specified point
  \param p point
  \param scrollMode scrolling mode
  \param direction direction
*/
void QwtWheel::getScrollMode( const QPoint &p, 
    QwtAbstractSlider::ScrollMode &scrollMode, int &direction ) const
{
    if ( wheelRect().contains( p ) )
        scrollMode = QwtAbstractSlider::ScrMouse;
    else
        scrollMode = QwtAbstractSlider::ScrNone;

    direction = 0;
}

/*!
  \brief Set the mass of the wheel

  Assigning a mass turns the wheel into a flywheel.
  \param mass The wheel's mass
*/
void QwtWheel::setMass( double mass )
{
    QwtAbstractSlider::setMass( mass );
}

/*!
  \brief Set the width of the wheel

  Corresponds to the wheel height for horizontal orientation,
  and the wheel width for vertical orientation.

  \param width the wheel's width
  \sa wheelWidth()
*/
void QwtWheel::setWheelWidth( int width )
{
    d_data->wheelWidth = width;
    update();
}

/*!
  \return Width of the wheel
  \sa setWheelWidth()
*/
int QwtWheel::wheelWidth() const
{
    return d_data->wheelWidth;
}

/*!
  \return a size hint
*/
QSize QwtWheel::sizeHint() const
{
    const QSize hint = minimumSizeHint();
    return hint.expandedTo( QApplication::globalStrut() );
}

/*!
  \brief Return a minimum size hint
  \warning The return value is based on the wheel width.
*/
QSize QwtWheel::minimumSizeHint() const
{
    QSize sz( 3 * d_data->wheelWidth + 2 * d_data->borderWidth,
        d_data->wheelWidth + 2 * d_data->borderWidth );
    if ( orientation() != Qt::Horizontal )
        sz.transpose();

    return sz;
}
