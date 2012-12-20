#ifndef __CLASSIFIER_H
#define __CLASSIFIER_H

#include <QLabel>
#include <QWidget>
#include <QString>
#include <openbr_plugin.h>

namespace br
{

class BR_EXPORT_GUI Classifier : public QLabel
{
    Q_OBJECT
    QString algorithm;

public:
    explicit Classifier(QWidget *parent = 0);
    void setAlgorithm(const QString &algorithm);

public slots:
    void classify(const br::File &file);

private slots:
    void setClassification(const QString &key, const QString &value);

private:
    void _classify(br::File file);

signals:
    void newClassification(QString key, QString value);
};

} // namespace br

#endif // __CLASSIFIER_H
