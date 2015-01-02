#ifndef BR_GUICLASSIFIER_H
#define BR_GUICLASSIFIER_H

#include <QLabel>
#include <QWidget>
#include <QString>
#include <openbr/openbr_plugin.h>

namespace br
{

class BR_EXPORT GUIClassifier : public QLabel
{
    Q_OBJECT
    QString algorithm;

public:
    explicit GUIClassifier(QWidget *parent = 0);
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

#endif // BR_GUICLASSIFIER_H
