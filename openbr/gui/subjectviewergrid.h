#ifndef SUBJECTVIEWERGRID_H
#define SUBJECTVIEWERGRID_H

#include <QObject>
#include <QGridLayout>
#include <QSharedPointer>

#include <openbr/openbr_plugin.h>

#include "subjectviewer.h"

namespace br
{

class BR_EXPORT SubjectViewerGrid : public QWidget
{
    Q_OBJECT

    QGridLayout gridLayout;
    QList< QSharedPointer<SubjectViewer> >subjectViewers;

public:
    explicit SubjectViewerGrid(QWidget *parent = 0);

public slots:
    void setFiles(const QList<br::FileList> &file);

signals:
    void newInput(br::File);
    void newInput(QImage);
    void newMousePoint(QPointF);
    void selectedInput(br::File);
};

}

#endif // SUBJECTVIEWERGRID_H
