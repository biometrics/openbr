#include <opencv2/core/core.hpp>
#include <qwt_picker_machine.h>
#include <qwt_scale_engine.h>
#include <qwt_series_data.h>
#include <qwt_text.h>

#include "scoredistribution.h"

/**** SCORE_DISTRIBUTION ****/
/*** PUBLIC ***/
ScoreDistribution::ScoreDistribution(QWidget *parent)
    : QwtPlot(parent)
    , qppPicker(canvas())
{
    axisScaleEngine(QwtPlot::xBottom)->setAttribute(QwtScaleEngine::Floating);
    enableAxis(QwtPlot::xBottom, false);
    enableAxis(QwtPlot::yLeft, false);
    setAutoReplot(true);
    setFrameShadow(QFrame::Sunken);
    setFrameStyle(QFrame::StyledPanel);

    QPen impostorPen; impostorPen.setColor(QColor(228, 26, 28, 240)); impostorPen.setWidth(2);
    qpcImpostor.setPen(impostorPen);
    qpcImpostor.setBrush(QColor(228, 26, 28, 180));
    qpcImpostor.setRenderHint(QwtPlotItem::RenderAntialiased);
    qpcImpostor.attach(this);

    QPen genuinePen; genuinePen.setColor(QColor(55, 126, 184, 240)); genuinePen.setWidth(2);
    qpcGenuine.setPen(genuinePen);
    qpcGenuine.setBrush(QColor(55, 126, 184, 180));
    qpcGenuine.setRenderHint(QwtPlotItem::RenderAntialiased);
    qpcGenuine.attach(this);

    QPen livePen; livePen.setColor(QColor(152, 78, 163, 200)); livePen.setWidth(4);
    qpcLive.setPen(livePen);
    qpcLive.setStyle(QwtPlotCurve::Dots);
    qpcLive.setRenderHint(QwtPlotItem::RenderAntialiased);
    qpcLive.attach(this);

    QPen pen; pen.setColor(QColor(77, 175, 74)); pen.setWidth(2);
    qpmMarker.setLinePen(pen);
    qpmMarker.setLineStyle(QwtPlotMarker::VLine);
    qpmMarker.setRenderHint(QwtPlotItem::RenderAntialiased);
    qpmMarker.attach(this);

    qppPicker.setStateMachine(new QwtPickerDragPointMachine());
    connect(&qppPicker, SIGNAL(selected(QPointF)), this, SLOT(selected(QPointF)));
    connect(&qppPicker, SIGNAL(moved(QPointF)), this, SLOT(selected(QPointF)));
}

/*** PUBLIC SLOTS ***/
void ScoreDistribution::setDistribution(const QString &distribution)
{
    QFile file(distribution);
    file.open(QFile::ReadOnly);
    QStringList lines = QString(file.readAll()).split('\n');
    file.close();

    QList<QPointF> genuinePoints, impostorPoints;
    foreach (const QString &line, lines) {
        if (!line.startsWith("KDE")) continue;
        QStringList words = line.split(',');
        const QPointF point(words[1].toFloat(), words[2].toFloat());
        if (line.startsWith("KDEGenuine")) genuinePoints.append(point);
        else                               impostorPoints.append(point);
    }

    // Downsample the points for displaying
    const float threshold = 0.01;
    for (int i=genuinePoints.size()-1; i>=1; i--)
        if ((genuinePoints[i] - genuinePoints[i-1]).manhattanLength() < threshold)
            genuinePoints.removeAt(i-1);
    for (int i=impostorPoints.size()-1; i>=1; i--)
        if ((impostorPoints[i] - impostorPoints[i-1]).manhattanLength() < threshold)
            impostorPoints.removeAt(i-1);

    qpcGenuine.setData(new QwtPointSeriesData(genuinePoints.toVector()));
    qpcImpostor.setData(new QwtPointSeriesData(impostorPoints.toVector()));
}

void ScoreDistribution::setThreshold(float score)
{
    qpmMarker.setXValue(score);
}

void ScoreDistribution::setLiveScores(const QList<float> &scores)
{
    QVector<QPointF> livePoints;
    foreach (float score, scores)
        livePoints.append(QPointF(score, cv::theRNG().uniform(0.f, 1.f)));
    qpcLive.setData(new QwtPointSeriesData(livePoints));
}

/*** PRIVATE SLOTS ***/
void ScoreDistribution::selected(const QPointF &point)
{
    if ((qpmMarker.xValue() == point.x()) ||
        (qpcLive.data()->size() <= 1))
        return;

    setThreshold(point.x());
    emit newThreshold(point.x());
}

#include "moc_scoredistribution.cpp"
