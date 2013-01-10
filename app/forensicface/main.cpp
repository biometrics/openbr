#include <QApplication>
#include <openbr.h>

#include "openbr-gui/initialize.h"
#include "openbr-gui/splashscreen.h"
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    br::SplashScreen splashScreen;
    splashScreen.show();

    br_initialize_gui();
    MainWindow mainWindow;
    mainWindow.show();
    splashScreen.finish(&mainWindow);

    int result = a.exec();
    br_finalize();
    return result;
}
