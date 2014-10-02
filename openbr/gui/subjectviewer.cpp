#include "subjectviewer.h"

using namespace br;

SubjectViewer::SubjectViewer(QWidget *parent)
    : QWidget(parent)
{
    files = FileList();
    currentIndex = 0;

    info.setText("-");
    info.setAlignment(Qt::AlignCenter);
    info.setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Maximum);

    layout.addWidget(&viewer);
    layout.addWidget(&info);

    setLayout(&layout);

    connect(&viewer, SIGNAL(newInput(br::File)), this, SIGNAL(newInput(br::File)));
    connect(&viewer, SIGNAL(newInput(QImage)), this, SIGNAL(newInput(QImage)));
    connect(&viewer, SIGNAL(newMousePoint(QPointF)), this, SIGNAL(newMousePoint(QPointF)));
    connect(&viewer, SIGNAL(selectedInput(br::File)), this, SIGNAL(selectedInput(br::File)));
}

void SubjectViewer::setFiles(const FileList &files)
{
    currentIndex = 0;
    this->files = files;

    if (!files.empty()) {
        info.setText(QString::number(currentIndex+1) + "/" + QString::number(files.size()));
        viewer.setFile(this->files[currentIndex]);
    } else {
        info.setText("-");
        viewer.setFile(File());
    }
}

void SubjectViewer::wheelEvent(QWheelEvent *event)
{
    QPoint numDegrees = event->angleDelta();

    if (numDegrees.y() < 0 && currentIndex > 0) {
        viewer.setFile(this->files[--currentIndex]);
        info.setText(QString::number(currentIndex+1) + "/" + QString::number(files.size()));
    } else if (numDegrees.y() > 0 && currentIndex < files.size()-1) {
        viewer.setFile(this->files[++currentIndex]);
        info.setText(QString::number(currentIndex+1) + "/" + QString::number(files.size()));
    }

    event->accept();
}
