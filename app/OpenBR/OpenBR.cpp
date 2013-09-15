#include <QApplication>
#include <QGridLayout>
#include <QLabel>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <openbr/openbr_plugin.h>
#include <openbr/gui/algorithm.h>
#include <openbr/gui/tail.h>
#include <openbr/gui/templateviewer.h>

using namespace br;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0)
        : QMainWindow(parent)
    {
        QGridLayout *gridLayout = new QGridLayout();
        Algorithm *algorithm = new Algorithm();
        algorithm->addAlgorithm("FaceRecognition", "Face Recognition");
        algorithm->addAlgorithm("PP5", "PittPatt");
        TemplateViewer *target = new TemplateViewer();
        TemplateViewer *query = new TemplateViewer();
        Tail *tail = new Tail();
        gridLayout->addWidget(algorithm, 0, 0, 1, 2);
        gridLayout->addWidget(query, 1, 0, 1, 1);
        gridLayout->addWidget(target, 1, 1, 1, 1);
        gridLayout->addWidget(tail, 2, 0, 1, 2);

        QMenuBar *menuBar = new QMenuBar();
        QMenu *helpMenu = new QMenu("Help");
        QAction *aboutAction = new QAction("About", this);
        QAction *contactAction = new QAction("Contact", this);
        helpMenu->addAction(aboutAction);
        helpMenu->addAction(contactAction);
        connect(aboutAction, SIGNAL(triggered()), this, SLOT(about()));
        connect(contactAction, SIGNAL(triggered()), this, SLOT(contact()));
        menuBar->addMenu(helpMenu);

        setGeometry(0, 0, 800, 600);
        setMenuBar(menuBar);
        setWindowIcon(QIcon(":/openbr.png"));
        setWindowTitle("OpenBR");
        setCentralWidget(new QWidget(this));
        centralWidget()->setLayout(gridLayout);

        connect(target, SIGNAL(newInput(File)), tail, SLOT(setTargetGallery(File)));
        connect(query, SIGNAL(newInput(File)), tail, SLOT(setQueryGallery(File)));
        connect(tail, SIGNAL(newTargetFile(File)), target, SLOT(setFile(File)));
        connect(tail, SIGNAL(newQueryFile(File)), query, SLOT(setFile(File)));
    }

private slots:
    void about()
    {
        QMessageBox::about(this, "About", Context::about());
    }

    void contact()
    {
        QMessageBox::about(this, "Contact", "openbr-dev@googlegroups.com\n\nPlease reach out to us with questions and comments on this public mailing list!");
    }
};

int main(int argc, char *argv[])
{
    QApplication application(argc, argv);
    Context::initialize(argc, argv);

    MainWindow mainWindow;
    mainWindow.show();

    const int result = application.exec();
    Context::finalize();
    return result;
}

#include "OpenBR.moc"
