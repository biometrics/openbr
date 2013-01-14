#include <qapplication.h>
#include "tvplot.h"

int main(int argc, char **argv)
{
    QApplication a(argc, argv);

    TVPlot plot;
    
    plot.resize(600,400);
    plot.show();

    return a.exec(); 
}
