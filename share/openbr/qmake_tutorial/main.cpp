#include <QApplication>
#include <QLabel>
#include <openbr/openbr.h>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    br_initialize(argc, argv);

    QLabel label;
    label.setText("Hello OpenBR!");
    label.show();

    int result = app.exec();
    br_finalize();
    return result;
}
