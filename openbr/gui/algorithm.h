#ifndef BR_ALGORITHM_H
#define BR_ALGORITHM_H

#include <QAction>
#include <QString>
#include <QMenu>
#include <openbr/openbr_export.h>

namespace br
{

class BR_EXPORT Algorithm : public QMenu
{
    Q_OBJECT
    QHash<QString, QString> displayNames;

public:
    explicit Algorithm(QWidget *parent = 0);

public slots:
    bool addAlgorithm(const QString &algorithm, const QString &displayName = "");

private slots:
    void setAlgorithm(QAction *action);

signals:
    void newAlgorithm(QString algorithm);
};

}

#endif // BR_ALGORITHM_H
