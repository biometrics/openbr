/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_WHEEL_H
#define QWT_WHEEL_H

#include "qwt_global.h"
#include "qwt_abstract_slider.h"

/*!
  \brief The Wheel Widget

  The wheel widget can be used to change values over a very large range
  in very small steps. Using the setMass member, it can be configured
  as a flywheel.

  \sa The radio example.
*/
class QWT_EXPORT QwtWheel : public QwtAbstractSlider
{
    Q_OBJECT

    Q_PROPERTY( double totalAngle READ totalAngle WRITE setTotalAngle )
    Q_PROPERTY( double viewAngle READ viewAngle WRITE setViewAngle )
    Q_PROPERTY( int tickCnt READ tickCnt WRITE setTickCnt )
    Q_PROPERTY( int wheelWidth READ wheelWidth WRITE setWheelWidth )
    Q_PROPERTY( int borderWidth READ borderWidth WRITE setBorderWidth )
    Q_PROPERTY( int wheelBorderWidth READ wheelBorderWidth WRITE setWheelBorderWidth )
    Q_PROPERTY( double mass READ mass WRITE setMass )

public:
    explicit QwtWheel( QWidget *parent = NULL );
    virtual ~QwtWheel();

public Q_SLOTS:
    void setTotalAngle ( double );
    void setViewAngle( double );

public:
    virtual void setOrientation( Qt::Orientation );

    double totalAngle() const;
    double viewAngle() const;

    void setTickCnt( int );
    int tickCnt() const;

    void setMass( double );
    double mass() const;

    void setWheelWidth( int );
    int wheelWidth() const;

    void setWheelBorderWidth( int );
    int wheelBorderWidth() const;

    void setBorderWidth( int );
    int borderWidth() const;

    QRect wheelRect() const;

    virtual QSize sizeHint() const;
    virtual QSize minimumSizeHint() const;

protected:
    virtual void paintEvent( QPaintEvent * );
    virtual void resizeEvent( QResizeEvent * );

    virtual void drawTicks( QPainter *, const QRectF & );
    virtual void drawWheelBackground( QPainter *, const QRectF & );

    virtual void valueChange();

    virtual double getValue( const QPoint & );
    virtual void getScrollMode( const QPoint &,
        QwtAbstractSlider::ScrollMode &, int &direction ) const;

private:
    class PrivateData;
    PrivateData *d_data;
};

#endif
