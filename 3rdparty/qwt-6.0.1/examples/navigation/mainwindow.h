#ifndef _MAINWINDOW_H_
#define _MAINWINDOW_H_ 1

#include <qwidget.h>
#include <qmainwindow.h>

class QwtPlotRescaler;
class QLabel;
class Plot;

class MainWindow: public QMainWindow
{
    Q_OBJECT

public:
    enum MouseMode
    {
        Tracking,
        Zooming,
        Panning
    };

    enum RescaleMode
    {
        KeepScales,
        Fixed,
        Expanding,
        Fitting
    };

    MainWindow();

private Q_SLOTS:
    void setMouseMode(int);
    void setRescaleMode(int);
    void showRatio(double, double);

private:
    QWidget *createPanel(QWidget *);
    Plot *createPlot(QWidget *);

    QwtPlotRescaler *d_rescaler;
    QLabel *d_navigationInfo;
    QLabel *d_rescaleInfo;

    Plot *d_plot;
};

#endif
