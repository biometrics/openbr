#include "subjectviewergrid.h"

using namespace br;

SubjectViewerGrid::SubjectViewerGrid(QWidget *parent) :
    QWidget(parent)
{
    setLayout(&gridLayout);
    setFiles(QList<FileList>());
}

void SubjectViewerGrid::setFiles(const QList<FileList> &files)
{
    const int size = std::max(1, (int)ceil(sqrt((float)files.size())));
    while (subjectViewers.size() < size*size) {
        subjectViewers.append(QSharedPointer<SubjectViewer>(new SubjectViewer()));
        connect(subjectViewers.last().data(), SIGNAL(newInput(br::File)), this, SIGNAL(newInput(br::File)));
        connect(subjectViewers.last().data(), SIGNAL(newInput(QImage)), this, SIGNAL(newInput(QImage)));
        connect(subjectViewers.last().data(), SIGNAL(newMousePoint(QPointF)), this, SIGNAL(newMousePoint(QPointF)));
        connect(subjectViewers.last().data(), SIGNAL(selectedInput(br::File)), this, SIGNAL(selectedInput(br::File)));
    }

    { // Clear layout
        QLayoutItem *child;
        while ((child = gridLayout.takeAt(0)) != 0)
            child->widget()->setVisible(false);
    }

    for (int i=0; i<subjectViewers.size(); i++) {
        if (i < size*size) {
            gridLayout.addWidget(subjectViewers[i].data(), i/size, i%size, 1, 1);
            subjectViewers[i]->setVisible(true);
        }

        if (i < files.size()) {
            subjectViewers[i]->setFiles(files[i]);
        } else {
            subjectViewers[i]->viewer.setDefaultText("<b>"+ (size > 1 ? QString() : QString("Drag Photo or Folder Here")) +"</b>");
            subjectViewers[i]->setFiles(FileList());
        }

        subjectViewers[i]->viewer.setEditable(files.size() == 1);
    }
}
