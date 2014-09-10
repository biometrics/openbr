#ifndef SUBJECTVIEWER_H
#define SUBJECTVIEWER_H

#include <QList>

#include <openbr/openbr_plugin.h>
#include <openbr/gui/templateviewer.h>

namespace br
{

class BR_EXPORT SubjectViewer : public TemplateViewer
{
    FileList files;
    int currentIndex;

public:
    explicit SubjectViewer(QWidget *parent = 0);

public slots:
    void setFiles(const FileList &files);

protected slots:
    void wheelEvent(QWheelEvent *);
};

}

#endif // SUBJECTVIEWER_H
