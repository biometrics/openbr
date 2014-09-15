#ifndef BR_TEMPLATEVIEWERGRID_H
#define BR_TEMPLATEVIEWERGRID_H

#include <QObject>
#include <QGridLayout>
#include <QList>
#include <QSharedPointer>

#include <openbr/openbr_plugin.h>

#include "templateviewer.h"

namespace br
{

class BR_EXPORT TemplateViewerGrid : public QWidget
{
    Q_OBJECT

    QGridLayout gridLayout;
    QList< QSharedPointer<TemplateViewer> > templateViewers;

public:
    explicit TemplateViewerGrid(QWidget *parent = 0);

public slots:
    void setFiles(const br::FileList &file);
    void setFormat(const QString &format);
    void setMousePoint(const QPointF &mousePoint);

signals:
    void newInput(br::File);
    void newInput(QImage);
    void newMousePoint(QPointF);
    void selectedInput(br::File);
};

} // namespace br

#endif // BR_TEMPLATEVIEWERGRID_H
