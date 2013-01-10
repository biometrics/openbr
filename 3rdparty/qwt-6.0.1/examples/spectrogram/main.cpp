#include <qapplication.h>
#include <qmainwindow.h>
#include <qtoolbar.h>
#include <qtoolbutton.h>
#include "plot.h"

class MainWindow: public QMainWindow
{
public:
    MainWindow(QWidget * = NULL);

private:
    Plot *d_plot;
};

MainWindow::MainWindow(QWidget *parent):
    QMainWindow(parent)
{
    d_plot = new Plot(this);

    setCentralWidget(d_plot);

    QToolBar *toolBar = new QToolBar(this);

    QToolButton *btnSpectrogram = new QToolButton(toolBar);
    btnSpectrogram->setText("Spectrogram");
    btnSpectrogram->setCheckable(true);
    btnSpectrogram->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    toolBar->addWidget(btnSpectrogram);
    connect(btnSpectrogram, SIGNAL(toggled(bool)), 
        d_plot, SLOT(showSpectrogram(bool)));

    QToolButton *btnContour = new QToolButton(toolBar);
    btnContour->setText("Contour");
    btnContour->setCheckable(true);
    btnContour->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    toolBar->addWidget(btnContour);
    connect(btnContour, SIGNAL(toggled(bool)), 
        d_plot, SLOT(showContour(bool)));

#ifndef QT_NO_PRINTER
    QToolButton *btnPrint = new QToolButton(toolBar);
    btnPrint->setText("Print");
    btnPrint->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    toolBar->addWidget(btnPrint);
    connect(btnPrint, SIGNAL(clicked()), 
        d_plot, SLOT(printPlot()) );
#endif

    addToolBar(toolBar);

    btnSpectrogram->setChecked(true);
    btnContour->setChecked(false);
}

int main(int argc, char **argv)
{
    QApplication a(argc, argv);

    MainWindow mainWindow;
    mainWindow.resize(600,400);
    mainWindow.show();

    return a.exec(); 
}
