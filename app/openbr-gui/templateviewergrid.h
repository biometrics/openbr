#ifndef __TEMPLATE_VIEWER_GRID_H
#define __TEMPLATE_VIEWER_GRID_H

#include <QGridLayout>
#include <QList>
#include <QSharedPointer>
#include <openbr_plugin.h>

#include "templateviewer.h"

namespace br
{

class BR_EXPORT_GUI TemplateViewerGrid : public QWidget
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
    void newInput(br::File input);
    void newInput(QImage input);
    void newMousePoint(QPointF mousePoint);
    void selectedInput(br::File input);
};

} // namespace br

#endif // __TEMPLATE_VIEWER_GRID_H
