/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_PLOT_RENDERER_H
#define QWT_PLOT_RENDERER_H

#include "qwt_global.h"
#include <qobject.h>

class QwtPlot;
class QwtScaleMap;
class QSizeF;
class QRectF;
class QPainter;
class QPaintDevice;

#ifndef QT_NO_PRINTER
class QPrinter;
#endif

#ifndef QWT_NO_SVG
#ifdef QT_SVG_LIB
class QSvgGenerator;
#endif
#endif

/*!
    \brief Renderer for exporting a plot to a document, a printer
           or anything else, that is supported by QPainter/QPaintDevice
*/
class QWT_EXPORT QwtPlotRenderer: public QObject
{
    Q_OBJECT

public:
    //! Disard flags
    enum DiscardFlag
    {
        //! Render all components of the plot
        DiscardNone             = 0x00,

        //! Don't render the background of the plot
        DiscardBackground       = 0x01,

        //! Don't render the title of the plot
        DiscardTitle            = 0x02,

        //! Don't render the legend of the plot
        DiscardLegend           = 0x04,

        //! Don't render the background of the canvas
        DiscardCanvasBackground = 0x08
    };

    //! Disard flags
    typedef QFlags<DiscardFlag> DiscardFlags;

    /*!
       \brief Layout flags
       \sa setLayoutFlag(), testLayoutFlag()
     */
    enum LayoutFlag
    {
        //! Use the default layout without margins and frames
        DefaultLayout   = 0x00,

        //! Render all frames of the plot
        KeepFrames      = 0x01,

        /*!
          Instead of the scales a box is painted around the plot canvas,
          where the scale ticks are aligned to.
         */
        FrameWithScales = 0x02
    };

    //! Layout flags
    typedef QFlags<LayoutFlag> LayoutFlags;

    explicit QwtPlotRenderer( QObject * = NULL );
    virtual ~QwtPlotRenderer();

    void setDiscardFlag( DiscardFlag flag, bool on = true );
    bool testDiscardFlag( DiscardFlag flag ) const;

    void setDiscardFlags( DiscardFlags flags );
    DiscardFlags discardFlags() const;

    void setLayoutFlag( LayoutFlag flag, bool on = true );
    bool testLayoutFlag( LayoutFlag flag ) const;

    void setLayoutFlags( LayoutFlags flags );
    LayoutFlags layoutFlags() const;

    void renderDocument( QwtPlot *, const QString &format,
        const QSizeF &sizeMM, int resolution = 85 );

    void renderDocument( QwtPlot *,
        const QString &title, const QString &format,
        const QSizeF &sizeMM, int resolution = 85 );

#ifndef QWT_NO_SVG
#ifdef QT_SVG_LIB
#if QT_VERSION >= 0x040500
    void renderTo( QwtPlot *, QSvgGenerator & ) const;
#endif
#endif
#endif

#ifndef QT_NO_PRINTER
    void renderTo( QwtPlot *, QPrinter & ) const;
#endif

    void renderTo( QwtPlot *, QPaintDevice &p ) const;

    virtual void render( QwtPlot *,
        QPainter *, const QRectF &rect ) const;

    virtual void renderLegendItem( const QwtPlot *, 
        QPainter *, const QWidget *, const QRectF & ) const;

    virtual void renderTitle( const QwtPlot *,
        QPainter *, const QRectF & ) const;

    virtual void renderScale( const QwtPlot *, QPainter *,
        int axisId, int startDist, int endDist,
        int baseDist, const QRectF & ) const;

    virtual void renderCanvas( const QwtPlot *,
        QPainter *, const QRectF &canvasRect,
        const QwtScaleMap* maps ) const;

    virtual void renderLegend( 
        const QwtPlot *, QPainter *, const QRectF & ) const;

protected:
    void buildCanvasMaps( const QwtPlot *,
        const QRectF &, QwtScaleMap maps[] ) const;

private:
    class PrivateData;
    PrivateData *d_data;
};

Q_DECLARE_OPERATORS_FOR_FLAGS( QwtPlotRenderer::DiscardFlags )
Q_DECLARE_OPERATORS_FOR_FLAGS( QwtPlotRenderer::LayoutFlags )

#endif
