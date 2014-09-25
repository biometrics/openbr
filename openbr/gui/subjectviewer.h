#ifndef SUBJECTVIEWER_H
#define SUBJECTVIEWER_H

#include <QObject>
#include <QSharedPointer>
#include <QList>
#include <QVBoxLayout>

#include <openbr/openbr_plugin.h>
#include <openbr/gui/templateviewer.h>

namespace br
{

class BR_EXPORT SubjectViewer : public QWidget
{
    Q_OBJECT

    QVBoxLayout layout;
    QLabel info;
    FileList files;
    int currentIndex;

public:
    explicit SubjectViewer(QWidget *parent = 0);

    TemplateViewer viewer;

public slots:
    void setFiles(const FileList &files);

protected slots:
    void wheelEvent(QWheelEvent *);

signals:
    void newInput(br::File);
    void newInput(QImage);
    void newMousePoint(QPointF);
    void selectedInput(br::File);
};

}

#endif // SUBJECTVIEWER_H
