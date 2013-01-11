/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_COMPASS_H
#define QWT_COMPASS_H 1

#include "qwt_global.h"
#include "qwt_dial.h"
#include <qstring.h>
#include <qmap.h>

class QwtCompassRose;

/*!
  \brief A Compass Widget

  QwtCompass is a widget to display and enter directions. It consists
  of a scale, an optional needle and rose.

  \image html dials1.png

  \note The examples/dials example shows how to use QwtCompass.
*/

class QWT_EXPORT QwtCompass: public QwtDial
{
    Q_OBJECT

public:
    explicit QwtCompass( QWidget* parent = NULL );
    virtual ~QwtCompass();

    void setRose( QwtCompassRose *rose );
    const QwtCompassRose *rose() const;
    QwtCompassRose *rose();

    const QMap<double, QString> &labelMap() const;
    QMap<double, QString> &labelMap();
    void setLabelMap( const QMap<double, QString> &map );

protected:
    virtual QwtText scaleLabel( double value ) const;

    virtual void drawRose( QPainter *, const QPointF &center,
        double radius, double north, QPalette::ColorGroup ) const;

    virtual void drawScaleContents( QPainter *,
        const QPointF &center, double radius ) const;

    virtual void keyPressEvent( QKeyEvent * );

private:
    void initCompass();

    class PrivateData;
    PrivateData *d_data;
};

#endif
