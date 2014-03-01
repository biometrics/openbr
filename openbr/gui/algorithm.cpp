#include <QStringList>
#include <openbr/openbr.h>
#include <openbr/openbr_plugin.h>

#include "algorithm.h"

/**** ALGORITHM ****/
/*** PUBLIC ***/
br::Algorithm::Algorithm(QWidget *parent)
    : QMenu(parent)
{
    setTitle("Algorithm");
    br_set_property("enrollAll", "true");
    connect(this, SIGNAL(triggered(QAction*)), this, SLOT(setAlgorithm(QAction*)));
}

/*** PUBLIC SLOTS ***/
bool br::Algorithm::addAlgorithm(const QString &algorithm, const QString &displayName)
{
    static QStringList availableAlgorithms;
    if (availableAlgorithms.isEmpty())
        availableAlgorithms = br::Context::objects("Abbreviation", ".*", false);

    if (!availableAlgorithms.contains(algorithm))
        return false;

    QString name;
    if (displayName.isEmpty()) {
        name = algorithm;
    } else {
        displayNames.insert(displayName, algorithm);
        name = displayName;
    }

    QAction *action = addAction(name);
    action->setCheckable(true);
    if (actions().size() == 1) action->trigger();
    return true;
}

/*** PRIVATE SLOTS ***/
void br::Algorithm::setAlgorithm(QAction *action)
{
    QString algorithm = action->text();
    if (displayNames.contains(algorithm))
        algorithm = displayNames[algorithm];

    br_set_property("algorithm", qPrintable(algorithm));
    emit newAlgorithm(algorithm);
}

#include "moc_algorithm.cpp"
