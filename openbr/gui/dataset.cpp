#include <QDir>
#include <QRegExp>
#include <QStringList>
#include <openbr/openbr.h>

#include "dataset.h"

/**** DATASET ****/
/*** PUBLIC ***/
br::Dataset::Dataset(QWidget *parent)
    : QComboBox(parent)
{
    setToolTip("Dataset");
    connect(this, SIGNAL(currentIndexChanged(QString)), this, SLOT(datasetChangedTo(QString)));
}

/*** PUBLIC SLOTS ***/
static bool compareDatasets(const QString &a, const QString &b)
{
    static QHash<QString,int> knownDatasets;
    if (knownDatasets.isEmpty()) {
        knownDatasets["MEDS"] = 0;
        knownDatasets["PCSO"] = 1;
        knownDatasets["Good"] = 2;
        knownDatasets["Bad"] = 3;
        knownDatasets["Ugly"] = 4;
    }

    if (!knownDatasets.contains(b)) return knownDatasets.contains(a);
    if (!knownDatasets.contains(a)) return false;
    return knownDatasets[a] < knownDatasets[b];
}

void br::Dataset::setAlgorithm(const QString &algorithm)
{
    this->algorithm = algorithm;
    QStringList datasets;
    QRegExp re("^" + algorithm + "_(.+).csv$");
    foreach (const QString &file, QDir(QString("%1/share/openbr/Algorithm_Dataset/").arg(br_sdk_path())).entryList())
        if (re.indexIn(file) != -1)
            datasets.append(re.cap(1));
    qSort(datasets.begin(), datasets.end(), compareDatasets);
    clear();
    addItems(datasets);
    setVisible(!datasets.isEmpty());
}

/*** PRIVATE SLOTS ***/
void br::Dataset::datasetChangedTo(const QString &dataset)
{
    emit newDataset(dataset);
    emit newDistribution(QString("%1/share/openbr/Algorithm_Dataset/%2_%3.csv").arg(br_sdk_path(), algorithm, dataset));
}

#include "moc_dataset.cpp"
