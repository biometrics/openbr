#include <QApplication>
#include <QMainWindow>
#include <openbr/openbr_plugin.h>

int main(int argc, char *argv[])
{
    QApplication application(argc, argv);
    br::Context::initialize(argc, argv);

    QMainWindow mainWindow;
    mainWindow.setWindowIcon(QIcon(":/openbr.png"));
    mainWindow.setWindowTitle("OpenBR");
    mainWindow.show();

    const int result = application.exec();
    br::Context::finalize();
    return result;
}
