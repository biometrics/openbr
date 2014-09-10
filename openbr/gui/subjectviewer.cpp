#include "subjectviewer.h"

using namespace br;

SubjectViewer::SubjectViewer(QWidget *parent)
    : TemplateViewer(parent)
{
    files = FileList();
    currentIndex = 0;
}

void SubjectViewer::setFiles(const FileList &files)
{
    if (files.isEmpty()) return;

    currentIndex = 0;
    this->files = files;
    TemplateViewer::setFile(this->files[currentIndex]);
}

void SubjectViewer::wheelEvent(QWheelEvent *event)
{
    QPoint numDegrees = event->angleDelta();

    if (numDegrees.y() < 0 && currentIndex > 0)
        TemplateViewer::setFile(this->files[--currentIndex]);
     else if (numDegrees.y() > 0 && currentIndex < files.size()-1)
        TemplateViewer::setFile(this->files[++currentIndex]);

    event->accept();
}
