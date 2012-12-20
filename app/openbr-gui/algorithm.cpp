#include <QStringList>
#include <openbr.h>

#include "algorithm.h"

/**** ALGORITHM ****/
/*** PUBLIC ***/
br::Algorithm::Algorithm(QWidget *parent)
    : QComboBox(parent)
{
    setToolTip("Algorithm");
    connect(this, SIGNAL(currentIndexChanged(QString)), this, SLOT(setAlgorithm(QString)));
}

/*** PUBLIC SLOTS ***/
bool br::Algorithm::addAlgorithm(const QString &algorithm, const QString &displayName)
{
    static QStringList availableAlgorithms;
    if (availableAlgorithms.isEmpty())
        availableAlgorithms = QString(br_objects("Abbreviation", ".*", false)).split("\n");

    if (!availableAlgorithms.contains(algorithm))
        return false;

    if (displayName.isEmpty()) {
        addItem(algorithm);
    } else {
        displayNames.insert(displayName, algorithm);
        addItem(displayName);
    }
    return true;
}

/*** PRIVATE SLOTS ***/
void br::Algorithm::setAlgorithm(QString algorithm)
{
    if (displayNames.contains(algorithm))
        algorithm = displayNames[algorithm];

    br_set_property("algorithm", qPrintable(algorithm));
    emit newAlgorithm(algorithm);
}

#include "moc_algorithm.cpp"
