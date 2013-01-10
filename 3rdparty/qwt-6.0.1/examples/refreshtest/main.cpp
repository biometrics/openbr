#include <qapplication.h>
#include "mainwindow.h"

int main(int argc, char **argv)
{
    QApplication a(argc, argv);

    MainWindow mainWindow;
    mainWindow.resize(600,400);
    mainWindow.show();

    return a.exec(); 
}
