#include "plot.h"
#include "curvedata.h"
#include "signaldata.h"
#include <qwt_plot_grid.h>
#include <qwt_plot_layout.h>
#include <qwt_plot_canvas.h>
#include <qwt_plot_marker.h>
#include <qwt_plot_curve.h>
#include <qwt_plot_directpainter.h>
#include <qwt_curve_fitter.h>
#include <qwt_painter.h>
#include <qevent.h>

Plot::Plot(QWidget *parent):
    QwtPlot(parent),
    d_paintedPoints(0),
    d_interval(0.0, 10.0),
    d_timerId(-1)
{
    d_directPainter = new QwtPlotDirectPainter();

    setAutoReplot(false);

    // The backing store is important, when working with widget
    // overlays ( f.e rubberbands for zooming ). 
    // Here we don't have them and the internal 
    // backing store of QWidget is good enough.

    canvas()->setPaintAttribute(QwtPlotCanvas::BackingStore, false);


#if defined(Q_WS_X11)
    // Even if not recommended by TrollTech, Qt::WA_PaintOutsidePaintEvent
    // works on X11. This has a nice effect on the performance.
    
    canvas()->setAttribute(Qt::WA_PaintOutsidePaintEvent, true);

    // Disabling the backing store of Qt improves the performance
    // for the direct painter even more, but the canvas becomes
    // a native window of the window system, receiving paint events
    // for resize and expose operations. Those might be expensive
    // when there are many points and the backing store of
    // the canvas is disabled. So in this application
    // we better don't both backing stores.

    if ( canvas()->testPaintAttribute( QwtPlotCanvas::BackingStore ) )
    {
        canvas()->setAttribute(Qt::WA_PaintOnScreen, true);
        canvas()->setAttribute(Qt::WA_NoSystemBackground, true);
    }

#endif

    initGradient();

    plotLayout()->setAlignCanvasToScales(true);

    setAxisTitle(QwtPlot::xBottom, "Time [s]");
    setAxisScale(QwtPlot::xBottom, d_interval.minValue(), d_interval.maxValue()); 
    setAxisScale(QwtPlot::yLeft, -200.0, 200.0);

    QwtPlotGrid *grid = new QwtPlotGrid();
    grid->setPen(QPen(Qt::gray, 0.0, Qt::DotLine));
    grid->enableX(true);
    grid->enableXMin(true);
    grid->enableY(true);
    grid->enableYMin(false);
    grid->attach(this);

    d_origin = new QwtPlotMarker();
    d_origin->setLineStyle(QwtPlotMarker::Cross);
    d_origin->setValue(d_interval.minValue() + d_interval.width() / 2.0, 0.0);
    d_origin->setLinePen(QPen(Qt::gray, 0.0, Qt::DashLine));
    d_origin->attach(this);

    d_curve = new QwtPlotCurve();
    d_curve->setStyle(QwtPlotCurve::Lines);
    d_curve->setPen(QPen(Qt::green));
#if 1
    d_curve->setRenderHint(QwtPlotItem::RenderAntialiased, true);
#endif
#if 1
    d_curve->setPaintAttribute(QwtPlotCurve::ClipPolygons, false);
#endif
    d_curve->setData(new CurveData());
    d_curve->attach(this);
}

Plot::~Plot()
{
    delete d_directPainter;
}

void Plot::initGradient()
{
    QPalette pal = canvas()->palette();

#if QT_VERSION >= 0x040400
    QLinearGradient gradient( 0.0, 0.0, 1.0, 0.0 );
    gradient.setCoordinateMode( QGradient::StretchToDeviceMode );
    gradient.setColorAt(0.0, QColor( 0, 49, 110 ) );
    gradient.setColorAt(1.0, QColor( 0, 87, 174 ) );

    pal.setBrush(QPalette::Window, QBrush(gradient));
#else
    pal.setBrush(QPalette::Window, QBrush( color ));
#endif

    canvas()->setPalette(pal);
}

void Plot::start()
{
    d_clock.start();
    d_timerId = startTimer(10);
}

void Plot::replot()
{
    CurveData *data = (CurveData *)d_curve->data();
    data->values().lock();

    QwtPlot::replot();
    d_paintedPoints = data->size();

    data->values().unlock();
}

void Plot::setIntervalLength(double interval)
{
    if ( interval > 0.0 && interval != d_interval.width() )
    {
        d_interval.setMaxValue(d_interval.minValue() + interval);
        setAxisScale(QwtPlot::xBottom, 
            d_interval.minValue(), d_interval.maxValue());

        replot();
    }
}

void Plot::updateCurve()
{
    CurveData *data = (CurveData *)d_curve->data();
    data->values().lock();

    const int numPoints = data->size();
    if ( numPoints > d_paintedPoints )
    {
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

            QRectF br = qwtBoundingRect( *data, 
                d_paintedPoints - 1, numPoints - 1 );

            const QRect clipRect = QwtScaleMap::transform( xMap, yMap, br ).toRect();
            d_directPainter->setClipRegion( clipRect );
        }

        d_directPainter->drawSeries(d_curve, 
            d_paintedPoints - 1, numPoints - 1);
        d_paintedPoints = numPoints;
    }

    data->values().unlock();
}

void Plot::incrementInterval()
{
    d_interval = QwtInterval(d_interval.maxValue(),
        d_interval.maxValue() + d_interval.width());

    CurveData *data = (CurveData *)d_curve->data();
    data->values().clearStaleValues(d_interval.minValue());

    // To avoid, that the grid is jumping, we disable 
    // the autocalculation of the ticks and shift them
    // manually instead.

    QwtScaleDiv scaleDiv = *axisScaleDiv(QwtPlot::xBottom);
    scaleDiv.setInterval(d_interval);

    for ( int i = 0; i < QwtScaleDiv::NTickTypes; i++ )
    {
        QList<double> ticks = scaleDiv.ticks(i);
        for ( int j = 0; j < ticks.size(); j++ )
            ticks[j] += d_interval.width();
        scaleDiv.setTicks(i, ticks);
    }
    setAxisScaleDiv(QwtPlot::xBottom, scaleDiv);

    d_origin->setValue(d_interval.minValue() + d_interval.width() / 2.0, 0.0);

    d_paintedPoints = 0;
    replot();
}

void Plot::timerEvent(QTimerEvent *event)
{
    if ( event->timerId() == d_timerId )
    {
        updateCurve();

        const double elapsed = d_clock.elapsed() / 1000.0;
        if ( elapsed > d_interval.maxValue() )
            incrementInterval();

        return;
    }

    QwtPlot::timerEvent(event);
}

void Plot::resizeEvent(QResizeEvent *event)
{
    d_directPainter->reset();
    QwtPlot::resizeEvent(event);
}

void Plot::showEvent( QShowEvent * )
{
    replot();
}
