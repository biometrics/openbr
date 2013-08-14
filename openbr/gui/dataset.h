#ifndef BR_DATASET_H
#define BR_DATASET_H

#include <QComboBox>
#include <QString>
#include <QWidget>
#include <openbr/openbr_export.h>

namespace br
{

class BR_EXPORT Dataset : public QComboBox
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

#endif // BR_DATASET_H
