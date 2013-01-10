#include <QFileInfo>
#include <QIcon>
#include <QPointF>
#include <openbr_plugin.h>

#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace br;

/**** MAIN_WINDOW ****/
/*** PUBLIC ***/
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowIcon(QIcon(":/icons/openbr.png"));
    setWindowTitle("Forensic Face " + Context::version());

    connect(&algorithm, SIGNAL(newAlgorithm(QString)), ui->tTail, SLOT(setAlgorithm(QString)));
    connect(ui->tTail, SIGNAL(newTargetFiles(br::FileList)), ui->gvTarget, SLOT(setFiles(br::FileList)));
    connect(ui->tTail, SIGNAL(newQueryFiles(br::FileList)), ui->gvQuery, SLOT(setFiles(br::FileList)));
    connect(ui->vView, SIGNAL(newFormat(QString)), &ui->gvTarget->tvgTemplateViewerGrid, SLOT(setFormat(QString)));
    connect(ui->vView, SIGNAL(newFormat(QString)), &ui->gvQuery->tvgTemplateViewerGrid, SLOT(setFormat(QString)));
    connect(ui->vView, SIGNAL(newCount(int)), ui->tTail, SLOT(setCount(int)));
    connect(&ui->gvTarget->gGallery, SIGNAL(newFiles(br::FileList)), ui->tTail, SLOT(setTargetGalleryFiles(br::FileList)));
    connect(&ui->gvQuery->gGallery, SIGNAL(newFiles(br::FileList)), ui->tTail, SLOT(setQueryGalleryFiles(br::FileList)));
    connect(&ui->gvTarget->gGallery, SIGNAL(newGallery(br::File)), ui->tTail, SLOT(setTargetGallery(br::File)));
    connect(&ui->gvQuery->gGallery, SIGNAL(newGallery(br::File)), ui->tTail, SLOT(setQueryGallery(br::File)));
    connect(&ui->gvTarget->tvgTemplateViewerGrid, SIGNAL(newMousePoint(QPointF)), &ui->gvQuery->tvgTemplateViewerGrid, SLOT(setMousePoint(QPointF)));
    connect(&ui->gvQuery->tvgTemplateViewerGrid, SIGNAL(newMousePoint(QPointF)), &ui->gvTarget->tvgTemplateViewerGrid, SLOT(setMousePoint(QPointF)));

    algorithm.addAlgorithm("OpenBR");
    algorithm.addAlgorithm("PP5");
    ui->tbAlgorithm->addWidget(&algorithm);
    ui->tbAlgorithm->setVisible(algorithm.count() > 1);

    setStatusBar(&progress);
}

MainWindow::~MainWindow()
{
    delete ui;
}
