/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_THERMO_H
#define QWT_THERMO_H

#include "qwt_global.h"
#include "qwt_abstract_scale.h"
#include "qwt_interval.h"
#include <qwidget.h>

class QwtScaleDraw;
class QwtColorMap;

/*!
  \brief The Thermometer Widget

  QwtThermo is a widget which displays a value in an interval. It supports:
  - a horizontal or vertical layout;
  - a range;
  - a scale;
  - an alarm level.

  \image html sysinfo.png

  The fill colors might be calculated from an optional color map
  If no color map has been assigned QwtThermo uses the 
  following colors/brushes from the widget palette:

  - QPalette::Base
    Background of the pipe
  - QPalette::ButtonText
    Fill brush below the alarm level
  - QPalette::Highlight
    Fill brush for the values above the alarm level
  - QPalette::WindowText
    For the axis of the scale
  - QPalette::Text
    For the labels of the scale

  By default, the scale and range run over the same interval of values.
  QwtAbstractScale::setScale() changes the interval of the scale and allows
  easy conversion between physical units.

  The example shows how to make the scale indicate in degrees Fahrenheit and
  to set the value in degrees Kelvin:
\code
#include <qapplication.h>
#include <qwt_thermo.h>

double Kelvin2Fahrenheit(double kelvin)
{
    // see http://en.wikipedia.org/wiki/Kelvin
    return 1.8*kelvin - 459.67;
}

int main(int argc, char **argv)
{
    const double minKelvin = 0.0;
    const double maxKelvin = 500.0;

    QApplication a(argc, argv);
    QwtThermo t;
    t.setRange(minKelvin, maxKelvin);
    t.setScale(Kelvin2Fahrenheit(minKelvin), Kelvin2Fahrenheit(maxKelvin));
    // set the value in Kelvin but the scale displays in Fahrenheit
    // 273.15 Kelvin = 0 Celsius = 32 Fahrenheit
    t.setValue(273.15);
    a.setMainWidget(&t);
    t.show();
    return a.exec();
}
\endcode

  \todo Improve the support for a logarithmic range and/or scale.
*/
class QWT_EXPORT QwtThermo: public QWidget, public QwtAbstractScale
{
    Q_OBJECT

    Q_ENUMS( ScalePos )

    Q_PROPERTY( bool alarmEnabled READ alarmEnabled WRITE setAlarmEnabled )
    Q_PROPERTY( double alarmLevel READ alarmLevel WRITE setAlarmLevel )
    Q_PROPERTY( ScalePos scalePosition READ scalePosition
        WRITE setScalePosition )
    Q_PROPERTY( int spacing READ spacing WRITE setSpacing )
    Q_PROPERTY( int borderWidth READ borderWidth WRITE setBorderWidth )
    Q_PROPERTY( double maxValue READ maxValue WRITE setMaxValue )
    Q_PROPERTY( double minValue READ minValue WRITE setMinValue )
    Q_PROPERTY( int pipeWidth READ pipeWidth WRITE setPipeWidth )
    Q_PROPERTY( double value READ value WRITE setValue )

public:
    /*!
      Scale position. QwtThermo tries to enforce valid combinations of its
      orientation and scale position:

      - Qt::Horizonal combines with NoScale, TopScale and BottomScale
      - Qt::Vertical combines with NoScale, LeftScale and RightScale

      \sa setOrientation(), setScalePosition()
    */
    enum ScalePos
    {
        //! No scale
        NoScale,

        //! The scale is left of the pipe
        LeftScale,

        //! The scale is right of the pipe
        RightScale,

        //! The scale is above the pipe
        TopScale,

        //! The scale is below the pipe
        BottomScale
    };

    explicit QwtThermo( QWidget *parent = NULL );
    virtual ~QwtThermo();

    void setOrientation( Qt::Orientation, ScalePos );

    void setScalePosition( ScalePos s );
    ScalePos scalePosition() const;

    void setSpacing( int );
    int spacing() const;

    void setBorderWidth( int w );
    int borderWidth() const;

    void setFillBrush( const QBrush &b );
    const QBrush &fillBrush() const;

    void setAlarmBrush( const QBrush &b );
    const QBrush &alarmBrush() const;

    void setAlarmLevel( double v );
    double alarmLevel() const;

    void setAlarmEnabled( bool tf );
    bool alarmEnabled() const;

    void setColorMap( QwtColorMap * );
    QwtColorMap *colorMap();
    const QwtColorMap *colorMap() const;

    void setPipeWidth( int w );
    int pipeWidth() const;

    void setRangeFlags( QwtInterval::BorderFlags );
    QwtInterval::BorderFlags rangeFlags() const;

    void setMaxValue( double v );
    double maxValue() const;

    void setMinValue( double v );
    double minValue() const;

    double value() const;

    void setRange( double vmin, double vmax, bool lg = false );

    virtual QSize sizeHint() const;
    virtual QSize minimumSizeHint() const;

    void setScaleDraw( QwtScaleDraw * );
    const QwtScaleDraw *scaleDraw() const;

public Q_SLOTS:
    virtual void setValue( double val );

protected:
    virtual void drawLiquid( QPainter *, const QRect & ) const;
    virtual void scaleChange();

    virtual void paintEvent( QPaintEvent * );
    virtual void resizeEvent( QResizeEvent * );
    virtual void changeEvent( QEvent * );

    QwtScaleDraw *scaleDraw();

    QRect pipeRect() const;

private:
    void layoutThermo( bool );

    class PrivateData;
    PrivateData *d_data;
};

#endif
