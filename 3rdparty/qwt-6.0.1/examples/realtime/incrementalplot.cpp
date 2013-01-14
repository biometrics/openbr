#include <qwt_plot.h>
#include <qwt_plot_canvas.h>
#include <qwt_plot_curve.h>
#include <qwt_symbol.h>
#include <qwt_plot_directpainter.h>
#include "incrementalplot.h"
#include <qpaintengine.h>

class CurveData: public QwtArraySeriesData<QPointF>
{
public:
    CurveData()
    {
    }

    virtual QRectF boundingRect() const
    {
        if ( d_boundingRect.width() < 0.0 )
            d_boundingRect = qwtBoundingRect( *this );

        return d_boundingRect;
    }

    inline void append( const QPointF &point )
    {
        d_samples += point;
    }

    void clear()
    {
        d_samples.clear();
        d_samples.squeeze();
        d_boundingRect = QRectF( 0.0, 0.0, -1.0, -1.0 );
    }
};

IncrementalPlot::IncrementalPlot(QWidget *parent): 
    QwtPlot(parent),
    d_curve(NULL)
{
    d_directPainter = new QwtPlotDirectPainter(this);

#if defined(Q_WS_X11)
    canvas()->setAttribute(Qt::WA_PaintOutsidePaintEvent, true);
    canvas()->setAttribute(Qt::WA_PaintOnScreen, true);
#endif

    d_curve = new QwtPlotCurve("Test Curve");
    d_curve->setStyle(QwtPlotCurve::NoCurve);
    d_curve->setData( new CurveData() );

    d_curve->setSymbol(new QwtSymbol(QwtSymbol::XCross,
        Qt::NoBrush, QPen(Qt::white), QSize( 4, 4 ) ) );

    d_curve->attach(this);

    setAutoReplot(false);
}

IncrementalPlot::~IncrementalPlot()
{
    delete d_curve;
}

void IncrementalPlot::appendPoint( const QPointF &point )
{
    CurveData *data = static_cast<CurveData *>( d_curve->data() );
    data->append(point);

    const bool doClip = !canvas()->testAttribute( Qt::WA_PaintOnScreen );
    if ( doClip )
    {
        /*
           Depending on the platform setting a clip might be an important
           performance issue. F.e. for Qt Embedded this reduces the
           part of the backing store that has to be copied out - maybe
           to an unaccelerated frame buffer device.
         */
        const QwtScaleMap xMap = canvasMap( d_curve->xAxis() );
        const QwtScaleMap yMap = canvasMap( d_curve->yAxis() );

        QRegion clipRegion;

        const QSize symbolSize = d_curve->symbol()->size();
        QRect r( 0, 0, symbolSize.width() + 2, symbolSize.height() + 2 );

        const QPointF center = 
            QwtScaleMap::transform( xMap, yMap, point );
        r.moveCenter( center.toPoint() );
        clipRegion += r;

        d_directPainter->setClipRegion( clipRegion );
    }
    
    d_directPainter->drawSeries(d_curve, 
        data->size() - 1, data->size() - 1);
}

void IncrementalPlot::clearPoints()
{
    CurveData *data = static_cast<CurveData *>( d_curve->data() );
    data->clear();

    replot();
}
