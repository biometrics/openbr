/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_SLIDER_H
#define QWT_SLIDER_H

#include "qwt_global.h"
#include "qwt_abstract_scale.h"
#include "qwt_abstract_slider.h"

class QwtScaleDraw;

/*!
  \brief The Slider Widget

  QwtSlider is a slider widget which operates on an interval
  of type double. QwtSlider supports different layouts as
  well as a scale.

  \image html sliders.png

  \sa QwtAbstractSlider and QwtAbstractScale for the descriptions
      of the inherited members.
*/

class QWT_EXPORT QwtSlider : public QwtAbstractSlider, public QwtAbstractScale
{
    Q_OBJECT
    Q_ENUMS( ScalePos )
    Q_ENUMS( BackgroundStyle )
    Q_PROPERTY( ScalePos scalePosition READ scalePosition
        WRITE setScalePosition )
    Q_PROPERTY( BackgroundStyles backgroundStyle 
        READ backgroundStyle WRITE setBackgroundStyle )
    Q_PROPERTY( QSize handleSize READ handleSize WRITE setHandleSize )
    Q_PROPERTY( int borderWidth READ borderWidth WRITE setBorderWidth )
    Q_PROPERTY( int spacing READ spacing WRITE setSpacing )

public:

    /*!
      Scale position. QwtSlider tries to enforce valid combinations of its
      orientation and scale position:

      - Qt::Horizonal combines with NoScale, TopScale and BottomScale
      - Qt::Vertical combines with NoScale, LeftScale and RightScale

      \sa QwtSlider()
     */
    enum ScalePos
    {
        //! The slider has no scale
        NoScale,

        //! The scale is left of the slider
        LeftScale,

        //! The scale is right of the slider
        RightScale,

        //! The scale is above of the slider
        TopScale,

        //! The scale is below of the slider
        BottomScale
    };

    /*!
      Background style.
      \sa QwtSlider()
     */
    enum BackgroundStyle
    {
        //! Trough background
        Trough = 0x01,

        //! Groove
        Groove = 0x02,
    };

    //! Background styles
    typedef QFlags<BackgroundStyle> BackgroundStyles;

    explicit QwtSlider( QWidget *parent,
        Qt::Orientation = Qt::Horizontal,
        ScalePos = NoScale, BackgroundStyles = Trough );

    virtual ~QwtSlider();

    virtual void setOrientation( Qt::Orientation );

    void setBackgroundStyle( BackgroundStyles );
    BackgroundStyles backgroundStyle() const;

    void setScalePosition( ScalePos s );
    ScalePos scalePosition() const;

    void setHandleSize( int width, int height );
    void setHandleSize( const QSize & );
    QSize handleSize() const;

    void setBorderWidth( int bw );
    int borderWidth() const;

    void setSpacing( int );
    int spacing() const;

    virtual QSize sizeHint() const;
    virtual QSize minimumSizeHint() const;

    void setScaleDraw( QwtScaleDraw * );
    const QwtScaleDraw *scaleDraw() const;

protected:
    virtual double getValue( const QPoint &p );
    virtual void getScrollMode( const QPoint &p,
        QwtAbstractSlider::ScrollMode &, int &direction ) const;

    virtual void drawSlider ( QPainter *, const QRect & ) const;
    virtual void drawHandle( QPainter *, const QRect &, int pos ) const;

    virtual void resizeEvent( QResizeEvent * );
    virtual void paintEvent ( QPaintEvent * );
    virtual void changeEvent( QEvent * );

    virtual void valueChange();
    virtual void rangeChange();
    virtual void scaleChange();

    int transform( double v ) const;

    QwtScaleDraw *scaleDraw();

private:
    void layoutSlider( bool );
    void initSlider( Qt::Orientation, ScalePos, BackgroundStyles );

    class PrivateData;
    PrivateData *d_data;
};

Q_DECLARE_OPERATORS_FOR_FLAGS( QwtSlider::BackgroundStyles )

#endif
