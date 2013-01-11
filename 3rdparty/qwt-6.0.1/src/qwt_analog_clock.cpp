/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_analog_clock.h"
#include <qmath.h>

/*!
  Constructor
  \param parent Parent widget
*/
QwtAnalogClock::QwtAnalogClock( QWidget *parent ):
    QwtDial( parent )
{
    initClock();
}

void QwtAnalogClock::initClock()
{
    setWrapping( true );
    setReadOnly( true );

    setOrigin( 270.0 );
    setRange( 0.0, 60.0 * 60.0 * 12.0 ); // seconds
    setScale( -1, 5, 60.0 * 60.0 );

    setScaleComponents( 
        QwtAbstractScaleDraw::Ticks | QwtAbstractScaleDraw::Labels );
    setScaleTicks( 1, 0, 8 );
    scaleDraw()->setSpacing( 8 );

    QColor knobColor = palette().color( QPalette::Active, QPalette::Text );
    knobColor = knobColor.dark( 120 );

    QColor handColor;
    int width;

    for ( int i = 0; i < NHands; i++ )
    {
        if ( i == SecondHand )
        {
            width = 2;
            handColor = knobColor.dark( 120 );
        }
        else
        {
            width = 8;
            handColor = knobColor;
        }

        QwtDialSimpleNeedle *hand = new QwtDialSimpleNeedle(
            QwtDialSimpleNeedle::Arrow, true, handColor, knobColor );
        hand->setWidth( width );

        d_hand[i] = NULL;
        setHand( ( Hand )i, hand );
    }
}

//! Destructor
QwtAnalogClock::~QwtAnalogClock()
{
    for ( int i = 0; i < NHands; i++ )
        delete d_hand[i];
}

/*!
  Nop method, use setHand instead
  \sa setHand()
*/
void QwtAnalogClock::setNeedle( QwtDialNeedle * )
{
    // no op
    return;
}

/*!
   Set a clockhand
   \param hand Specifies the type of hand
   \param needle Hand
   \sa hand()
*/
void QwtAnalogClock::setHand( Hand hand, QwtDialNeedle *needle )
{
    if ( hand >= 0 || hand < NHands )
    {
        delete d_hand[hand];
        d_hand[hand] = needle;
    }
}

/*!
  \return Clock hand
  \param hd Specifies the type of hand
  \sa setHand()
*/
QwtDialNeedle *QwtAnalogClock::hand( Hand hd )
{
    if ( hd < 0 || hd >= NHands )
        return NULL;

    return d_hand[hd];
}

/*!
  \return Clock hand
  \param hd Specifies the type of hand
  \sa setHand()
*/
const QwtDialNeedle *QwtAnalogClock::hand( Hand hd ) const
{
    return const_cast<QwtAnalogClock *>( this )->hand( hd );
}

/*!
  \brief Set the current time

  This is the same as QwtAnalogClock::setTime(), but Qt < 3.0
  can't handle default parameters for slots.
*/
void QwtAnalogClock::setCurrentTime()
{
    setTime( QTime::currentTime() );
}

/*!
  Set a time
  \param time Time to display
*/
void QwtAnalogClock::setTime( const QTime &time )
{
    if ( time.isValid() )
    {
        setValue( ( time.hour() % 12 ) * 60.0 * 60.0
            + time.minute() * 60.0 + time.second() );
    }
    else
        setValid( false );
}

/*!
  Find the scale label for a given value

  \param value Value
  \return Label
*/
QwtText QwtAnalogClock::scaleLabel( double value ) const
{
    if ( value == 0.0 )
        value = 60.0 * 60.0 * 12.0;

    return QString::number( int( value / ( 60.0 * 60.0 ) ) );
}

/*!
  \brief Draw the needle

  A clock has no single needle but three hands instead. drawNeedle
  translates value() into directions for the hands and calls
  drawHand().

  \param painter Painter
  \param center Center of the clock
  \param radius Maximum length for the hands
  \param dir Dummy, not used.
  \param colorGroup ColorGroup

  \sa drawHand()
*/
void QwtAnalogClock::drawNeedle( QPainter *painter, const QPointF &center,
    double radius, double dir, QPalette::ColorGroup colorGroup ) const
{
    Q_UNUSED( dir );

    if ( isValid() )
    {
        const double hours = value() / ( 60.0 * 60.0 );
        const double minutes = 
            ( value() - qFloor(hours) * 60.0 * 60.0 ) / 60.0;
        const double seconds = value() - qFloor(hours) * 60.0 * 60.0
            - qFloor(minutes) * 60.0;

        double angle[NHands];
        angle[HourHand] = 360.0 * hours / 12.0;
        angle[MinuteHand] = 360.0 * minutes / 60.0;
        angle[SecondHand] = 360.0 * seconds / 60.0;

        for ( int hand = 0; hand < NHands; hand++ )
        {
            double d = angle[hand];
            if ( direction() == Clockwise )
                d = 360.0 - d;

            d -= origin();

            drawHand( painter, ( Hand )hand, center, radius, d, colorGroup );
        }
    }
}

/*!
  Draw a clock hand

  \param painter Painter
  \param hd Specify the type of hand
  \param center Center of the clock
  \param radius Maximum length for the hands
  \param direction Direction of the hand in degrees, counter clockwise
  \param cg ColorGroup
*/
void QwtAnalogClock::drawHand( QPainter *painter, Hand hd,
    const QPointF &center, double radius, double direction,
    QPalette::ColorGroup cg ) const
{
    const QwtDialNeedle *needle = hand( hd );
    if ( needle )
    {
        if ( hd == HourHand )
            radius = qRound( 0.8 * radius );

        needle->draw( painter, center, radius, direction, cg );
    }
}
