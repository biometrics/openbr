#ifndef __DATASET_H
#define __DATASET_H

#include <QComboBox>
#include <QString>
#include <QWidget>
#include <openbr_export.h>

namespace br
{

class BR_EXPORT_GUI Dataset : public QComboBox
{
    Q_OBJECT
    QString algorithm;

public:
    explicit Dataset(QWidget *parent = 0);

public slots:
    void setAlgorithm(const QString &algorithm);

private slots:
    void datasetChangedTo(const QString &dataset);

signals:
    void newDataset(QString);
    void newDistribution(QString);
};

} // namespace br

#endif // __DATASET_H
