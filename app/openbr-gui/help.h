#ifndef HELP_H
#define HELP_H

#include <QAction>
#include <QList>
#include <QMenu>
#include <QSharedPointer>
#include <QWidget>
#include <openbr_export.h>

namespace br
{

class BR_EXPORT_GUI Help : public QMenu
{
    Q_OBJECT
    QList< QSharedPointer<QAction> > actions;

public:
    explicit Help(QWidget *parent = 0);

public slots:
    void showAbout();
    void showDocumentation();
    void showLicense();
};

} // namespace br

#endif // HELP_H
