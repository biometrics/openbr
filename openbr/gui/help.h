#ifndef BR_HELP_H
#define BR_HELP_H

#include <QAction>
#include <QList>
#include <QMenu>
#include <QSharedPointer>
#include <QWidget>
#include <openbr/openbr_export.h>

namespace br
{

class BR_EXPORT Help : public QMenu
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

#endif // BR_HELP_H
