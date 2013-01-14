#ifndef SCOREDISTRIBUTION_H
#define SCOREDISTRIBUTION_H

#include <QList>
#include <QKeyEvent>
#include <QPointF>
#include <QString>
#include <QWheelEvent>
#include <QWidget>
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_plot_marker.h>
#include <qwt_plot_picker.h>

class ScoreDistribution : public QwtPlot
{
    Q_OBJECT
    QwtPlotCurve qpcImpostor, qpcGenuine, qpcLive;
    QwtPlotMarker qpmMarker;
    QwtPlotPicker qppPicker;

public:
    explicit ScoreDistribution(QWidget *parent = 0);

public slots:
    void setDistribution(const QString &distribution);
    void setThreshold(float score);
    void setLiveScores(const QList<float> &scores);

private slots:
    void selected(const QPointF &point);

signals:
    void newThreshold(float);
};

#endif // SCOREDISTRIBUTION_H
