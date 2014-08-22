#include "formcombowidget.h"

using namespace br;

FormComboWidget::FormComboWidget(const QString &name, QWidget *parent) :
    QWidget(parent)
{
    combo = new QComboBox(this);
    form = new QFormLayout(this);

    form->addRow(name, combo);

    connect(combo, SIGNAL(activated(QString)), this, SIGNAL(activated(QString)));
    connect(combo, SIGNAL(currentIndexChanged(QString)), this, SIGNAL(currentIndexChanged(QString)));
}
