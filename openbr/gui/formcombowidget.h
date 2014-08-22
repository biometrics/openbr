#ifndef BR_FORMCOMBOWIDGET_H
#define BR_FORMCOMBOWIDGET_H

#include <QWidget>
#include <QFormLayout>
#include <QStringList>
#include <QComboBox>

#include <openbr/openbr_export.h>

namespace br {

class BR_EXPORT FormComboWidget : public QWidget
{
    Q_OBJECT

    QStringList items;
    QFormLayout *form;
    QComboBox *combo;

public:
    explicit FormComboWidget(const QString &name, QWidget *parent = 0);

public slots:
    void addItem(const QString &str) { items.append(str); combo->clear(); combo->addItems(items);  }

signals:
    void activated(QString);
    void currentIndexChanged(QString);
};

}

#endif // BR_FORMCOMBOWIDGET_H
