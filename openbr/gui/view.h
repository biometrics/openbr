#ifndef BR_VIEW_H
#define BR_VIEW_H

#include <QAction>
#include <QActionGroup>
#include <QLabel>
#include <QString>
#include <QToolBar>
#include <QToolButton>
#include <QWidget>
#include <openbr/openbr_export.h>

namespace br
{

class BR_EXPORT View : public QToolBar
{
    Q_OBJECT
    QToolButton tbPhoto, tbRegistered, tbEnhanced, tbFeatures;
    QToolButton tb1, tb4, tb9, tb16;
    QActionGroup agFormat, agCount;

public:
    explicit View(QWidget *parent = 0);

private slots:
    void formatChanged(QAction *action);
    void countChanged(QAction *action);

signals:
    void newFormat(QString image);
    void newCount(int count);
};

} // namespace br

#endif // BR_VIEW_H
