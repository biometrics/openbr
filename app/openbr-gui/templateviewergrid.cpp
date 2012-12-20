#include "templateviewergrid.h"

using namespace br;

/**** TEMPLATE_VIEWER_GRID ****/
/*** PUBLIC ***/
TemplateViewerGrid::TemplateViewerGrid(QWidget *parent)
    : QWidget(parent)
{
    setLayout(&gridLayout);
    setFiles(FileList(16));
    setFiles(FileList(1));
}

/*** PUBLIC SLOTS ***/
void TemplateViewerGrid::setFiles(const FileList &files)
{
    const int size = std::max(1, (int)ceil(sqrt((float)files.size())));
    while (templateViewers.size() < size*size) {
        templateViewers.append(QSharedPointer<TemplateViewer>(new TemplateViewer()));
        connect(templateViewers.last().data(), SIGNAL(newInput(br::File)), this, SIGNAL(newInput(br::File)));
        connect(templateViewers.last().data(), SIGNAL(newInput(QImage)), this, SIGNAL(newInput(QImage)));
        connect(templateViewers.last().data(), SIGNAL(newMousePoint(QPointF)), this, SIGNAL(newMousePoint(QPointF)));
        connect(templateViewers.last().data(), SIGNAL(selectedInput(br::File)), this, SIGNAL(selectedInput(br::File)));
    }

    { // Clear layout
        QLayoutItem *child;
        while ((child = gridLayout.takeAt(0)) != 0)
            child->widget()->setVisible(false);
    }

    for (int i=0; i<templateViewers.size(); i++) {
        if (i < size*size) {
            gridLayout.addWidget(templateViewers[i].data(), i/size, i%size, 1, 1);
            templateViewers[i]->setVisible(true);
        }

        if (i < files.size()) {
            templateViewers[i]->setFile(files[i]);
        } else {
            templateViewers[i]->setDefaultText("<b>"+ (size > 1 ? QString() : QString("Drag Photo or Folder Here")) +"</b>");
            templateViewers[i]->setFile(QString());
        }

        templateViewers[i]->setEditable(files.size() == 1);
    }
}

void TemplateViewerGrid::setFormat(const QString &format)
{
    foreach (const QSharedPointer<TemplateViewer> &templateViewer, templateViewers)
        templateViewer->setFormat(format);
}

void TemplateViewerGrid::setMousePoint(const QPointF &mousePoint)
{
    foreach (const QSharedPointer<TemplateViewer> &templateViewer, templateViewers)
        templateViewer->setMousePoint(mousePoint);
}

#include "moc_templateviewergrid.cpp"
