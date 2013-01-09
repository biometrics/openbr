/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_SYMBOL_H
#define QWT_SYMBOL_H

#include "qwt_global.h"
#include <QPolygonF>

class QPainter;
class QRect;
class QSize;
class QBrush;
class QPen;
class QColor;
class QPointF;

//! A class for drawing symbols
class QWT_EXPORT QwtSymbol
{
public:
    /*!
      Symbol Style
      \sa setStyle(), style()
     */
    enum Style
    {
        //! No Style. The symbol cannot be drawn.
        NoSymbol = -1,

        //! Ellipse or circle
        Ellipse,

        //! Rectangle
        Rect,

        //!  Diamond
        Diamond,

        //! Triangle pointing upwards
        Triangle,

        //! Triangle pointing downwards
        DTriangle,

        //! Triangle pointing upwards
        UTriangle,

        //! Triangle pointing left
        LTriangle,

        //! Triangle pointing right
        RTriangle,

        //! Cross (+)
        Cross,

        //! Diagonal cross (X)
        XCross,

        //! Horizontal line
        HLine,

        //! Vertical line
        VLine,

        //! X combined with +
        Star1,

        //! Six-pointed star
        Star2,

        //! Hexagon
        Hexagon,

        /*!
         Styles >= QwtSymbol::UserSymbol are reserved for derived
         classes of QwtSymbol that overload drawSymbols() with
         additional application specific symbol types.
         */
        UserStyle = 1000
    };

public:
    QwtSymbol( Style = NoSymbol );
    QwtSymbol( Style, const QBrush &, const QPen &, const QSize & );
    QwtSymbol( const QwtSymbol & );
    virtual ~QwtSymbol();

    QwtSymbol &operator=( const QwtSymbol & );
    bool operator==( const QwtSymbol & ) const;
    bool operator!=( const QwtSymbol & ) const;

    void setSize( const QSize & );
    void setSize( int width, int height = -1 );
    const QSize& size() const;

    virtual void setColor( const QColor & );

    void setBrush( const QBrush& b );
    const QBrush& brush() const;

    void setPen( const QPen & );
    const QPen& pen() const;

    void setStyle( Style );
    Style style() const;

    void drawSymbol( QPainter *, const QPointF & ) const;
    void drawSymbols( QPainter *, const QPolygonF & ) const;

    virtual QSize boundingSize() const;

protected:
    virtual void drawSymbols( QPainter *,
        const QPointF *, int numPoints ) const;

private:
    class PrivateData;
    PrivateData *d_data;
};

/*!
  \brief Draw the symbol at a specified position

  \param painter Painter
  \param pos Position of the symbol in screen coordinates
*/
inline void QwtSymbol::drawSymbol(
    QPainter *painter, const QPointF &pos ) const
{
    drawSymbols( painter, &pos, 1 );
}

/*!
  \brief Draw symbols at the specified points

  \param painter Painter
  \param points Positions of the symbols in screen coordinates
*/

inline void QwtSymbol::drawSymbols(
    QPainter *painter, const QPolygonF &points ) const
{
    drawSymbols( painter, points.data(), points.size() );
}

#endif
