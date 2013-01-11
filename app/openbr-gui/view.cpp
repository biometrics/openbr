#include "view.h"

using namespace br;

/**** VIEW ****/
/*** PUBLIC ***/
View::View(QWidget *parent)
    : QToolBar(parent)
    , agFormat(parent)
    , agCount(parent)
{
    agFormat.addAction("Photo");
    agFormat.actions().last()->setShortcut(Qt::ControlModifier + Qt::Key_D);
    agFormat.addAction("Registered");
    agFormat.actions().last()->setShortcut(Qt::ControlModifier + Qt::Key_R);
    agFormat.addAction("Enhanced");
    agFormat.actions().last()->setShortcut(Qt::ControlModifier + Qt::Key_E);
    agFormat.addAction("Features");
    agFormat.actions().last()->setShortcut(Qt::ControlModifier + Qt::Key_F);

    agCount.addAction("1");
    agCount.actions().last()->setShortcut(Qt::ControlModifier + Qt::Key_1);
    agCount.addAction("4");
    agCount.actions().last()->setShortcut(Qt::ControlModifier + Qt::Key_2);
    agCount.addAction("9");
    agCount.actions().last()->setShortcut(Qt::ControlModifier + Qt::Key_3);
    agCount.addAction("16");
    agCount.actions().last()->setShortcut(Qt::ControlModifier + Qt::Key_4);

    addActions(agFormat.actions());
    addSeparator();
    addActions(agCount.actions());

    setToolTip("View");

    connect(&agFormat, SIGNAL(triggered(QAction*)), this, SLOT(formatChanged(QAction*)));
    connect(&agCount, SIGNAL(triggered(QAction*)), this, SLOT(countChanged(QAction*)));

    foreach (QAction *action, agCount.actions())
        action->setCheckable(true);
    agCount.actions()[2]->setChecked(true);

    foreach (QAction *action, agFormat.actions())
        action->setCheckable(true);
    agFormat.actions()[1]->setChecked(true);
}

/*** PRIVATE SLOTS ***/
void View::formatChanged(QAction *action)
{
    emit newFormat(action->text());
}

void View::countChanged(QAction *action)
{
    emit newCount(action->text().toInt());
}

#include "moc_view.cpp"
