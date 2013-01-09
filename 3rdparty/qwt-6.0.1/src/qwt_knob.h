/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_KNOB_H
#define QWT_KNOB_H

#include "qwt_global.h"
#include "qwt_abstract_slider.h"
#include "qwt_abstract_scale.h"

class QwtRoundScaleDraw;

/*!
  \brief The Knob Widget

  The QwtKnob widget imitates look and behaviour of a volume knob on a radio.
  It contains a scale around the knob which is set up automatically or can
  be configured manually (see QwtAbstractScale).
  Automatic scrolling is enabled when the user presses a mouse
  button on the scale. For a description of signals, slots and other
  members, see QwtAbstractSlider.

  \image html knob.png
  \sa   QwtAbstractSlider and QwtAbstractScale for the descriptions
    of the inherited members.
*/

class QWT_EXPORT QwtKnob : public QwtAbstractSlider, public QwtAbstractScale
{
    Q_OBJECT

    Q_ENUMS ( KnobStyle )
    Q_ENUMS ( MarkerStyle )

    Q_PROPERTY( KnobStyle knobStyle READ knobStyle WRITE setKnobStyle )
    Q_PROPERTY( MarkerStyle markerStyle READ markerStyle WRITE setMarkerStyle )
    Q_PROPERTY( int knobWidth READ knobWidth WRITE setKnobWidth )
    Q_PROPERTY( int borderWidth READ borderWidth WRITE setBorderWidth )
    Q_PROPERTY( double totalAngle READ totalAngle WRITE setTotalAngle )
    Q_PROPERTY( int markerSize READ markerSize WRITE setMarkerSize )
    Q_PROPERTY( int borderWidth READ borderWidth WRITE setBorderWidth )

public:
    /*! 
       \brief Style of the knob surface

       Depending on the KnobStyle the surface of the knob is
       filled from the brushes of the widget palette().

       \sa setKnobStyle(), knobStyle()
     */
    enum KnobStyle
    {
        //! Fill the knob with a brush from QPalette::Button.
        NoStyle = -1,

        //! Build a gradient from QPalette::Midlight and QPalette::Button
        Raised,

        /*! 
          Build a gradient from QPalette::Midlight, QPalette::Button
          and QPalette::Midlight
         */
        Sunken
    };

    /*!
        \brief Marker type
 
        The marker indicates the current value on the knob
        The default setting is a Notch marker.

        \sa setMarkerStyle(), setMarkerSize()
    */
    enum MarkerStyle 
    { 
        //! Don't paint any marker
        NoMarker = -1,

        //! Paint a single tick in QPalette::ButtonText color
        Tick, 

        //! Paint a circle in QPalette::ButtonText color
        Dot, 

        /*! 
          Draw a raised ellipse with a gradient build from
          QPalette::Light and QPalette::Mid
         */ 
        Nub, 

        /*! 
          Draw a sunken ellipse with a gradient build from
          QPalette::Light and QPalette::Mid
         */ 
        Notch 
    };

    explicit QwtKnob( QWidget* parent = NULL );
    virtual ~QwtKnob();

    void setKnobWidth( int w );
    int knobWidth() const;

    void setTotalAngle ( double angle );
    double totalAngle() const;

    void setKnobStyle( KnobStyle );
    KnobStyle knobStyle() const;

    void setBorderWidth( int bw );
    int borderWidth() const;

    void setMarkerStyle( MarkerStyle );
    MarkerStyle markerStyle() const;

    void setMarkerSize( int );
    int markerSize() const;

    virtual QSize sizeHint() const;
    virtual QSize minimumSizeHint() const;

    void setScaleDraw( QwtRoundScaleDraw * );
    const QwtRoundScaleDraw *scaleDraw() const;
    QwtRoundScaleDraw *scaleDraw();

protected:
    virtual void paintEvent( QPaintEvent * );
    virtual void resizeEvent( QResizeEvent * );
    virtual void changeEvent( QEvent * );

    virtual void drawKnob( QPainter *, const QRectF & ) const;
    virtual void drawMarker( QPainter *, 
        const QRectF &, double arc ) const;

    virtual double getValue( const QPoint &p );
    virtual void getScrollMode( const QPoint &, 
        QwtAbstractSlider::ScrollMode &, int &direction ) const;

private:
    void initKnob();
    void layoutKnob( bool update );
    void recalcAngle();

    virtual void valueChange();
    virtual void rangeChange();
    virtual void scaleChange();

    class PrivateData;
    PrivateData *d_data;
};

#endif
