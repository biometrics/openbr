#ifndef _TV_PLOT_H_

#include <qwt_plot.h>

class TVPlot: public QwtPlot
{
    Q_OBJECT

public:
    TVPlot(QWidget * = NULL);

private:
    void populate();

private Q_SLOTS:
    void showItem(QwtPlotItem *, bool on);
};

#endif
