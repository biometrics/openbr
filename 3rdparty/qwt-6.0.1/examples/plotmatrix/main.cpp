#include <qapplication.h>
#include <qpen.h>
#include <qwt_plot_grid.h>
#include "plotmatrix.h"

class MainWindow: public PlotMatrix
{
public:
    MainWindow();
};

MainWindow::MainWindow():
    PlotMatrix(3, 4)
{
    enableAxis(QwtPlot::yLeft);
    enableAxis(QwtPlot::yRight);
    enableAxis(QwtPlot::xBottom);

    for ( int row = 0; row < numRows(); row++ )
    {
        const double v = qPow(10.0, row);
        setAxisScale(QwtPlot::yLeft, row, -v, v);
        setAxisScale(QwtPlot::yRight, row, -v, v);
    }

    for ( int col = 0; col < numColumns(); col++ )
    {
        const double v = qPow(10.0, col);
        setAxisScale(QwtPlot::xBottom, col, -v, v);
        setAxisScale(QwtPlot::xTop, col, -v, v);
    }

    for ( int row = 0; row < numRows(); row++ )
    {
        for ( int col = 0; col < numColumns(); col++ )
        {
            QwtPlot *plt = plot(row, col);
            plt->setCanvasBackground(QColor(Qt::darkBlue));

            QwtPlotGrid *grid = new QwtPlotGrid();
            grid->enableXMin(true);
            grid->setMajPen(QPen(Qt::white, 0, Qt::DotLine));
            grid->setMinPen(QPen(Qt::gray, 0 , Qt::DotLine));
            grid->attach(plt);
        }
    }
}

int main(int argc, char **argv)
{
    QApplication a(argc, argv);

    MainWindow mainWindow;

    mainWindow.resize(800,600);
    mainWindow.show();

    return a.exec(); 
}
