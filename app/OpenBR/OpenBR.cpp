#include <QApplication>
#include <QGridLayout>
#include <QLabel>
#include <QMainWindow>
#include <openbr/openbr_plugin.h>
#include <openbr/gui/tail.h>
#include <openbr/gui/templateviewer.h>

using namespace br;

int main(int argc, char *argv[])
{
    QApplication application(argc, argv);
    Context::initialize(argc, argv);

    QGridLayout *gridLayout = new QGridLayout();
    TemplateViewer *target = new TemplateViewer();
    TemplateViewer *query = new TemplateViewer();
    Tail *tail = new Tail();
    gridLayout->addWidget(query, 0, 0, 1, 1);
    gridLayout->addWidget(target, 0, 1, 1, 1);
    gridLayout->addWidget(tail, 1, 0, 1, 2);

    QMainWindow mainWindow;
    mainWindow.setGeometry(0, 0, 800, 600);
    mainWindow.setWindowIcon(QIcon(":/openbr.png"));
    mainWindow.setWindowTitle("OpenBR");
    mainWindow.setCentralWidget(new QWidget(&mainWindow));
    mainWindow.centralWidget()->setLayout(gridLayout);
    mainWindow.show();

    const int result = application.exec();
    Context::finalize();
    return result;
}
