#include "rangewidget.h"

using namespace br;

RangeWidget::RangeWidget(QWidget *parent) :
    QWidget(parent)
{
    hLayout = new QHBoxLayout();

    validator = new QIntValidator(0,100,this);

    median = new QLineEdit("20", this);
    median->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    median->setAlignment(Qt::AlignCenter);
    median->setValidator(validator);

    rangeSpin = new QSpinBox(this);
    rangeSpin->setValue(0);
    rangeSpin->setAlignment(Qt::AlignCenter);

    label = new QLabel(this);
    label->setAlignment(Qt::AlignCenter);
    label->setText(QString::fromUtf8("±"));

    hLayout->addWidget(median);
    hLayout->addWidget(label);
    hLayout->addWidget(rangeSpin);
    hLayout->setSpacing(0);

    connect(median, SIGNAL(textChanged(QString)), this, SLOT(rangeChanged()));
    connect(rangeSpin, SIGNAL(valueChanged(int)), this, SLOT(rangeChanged()));

    vLayout.addLayout(hLayout);

    setLayout(&vLayout);
}

void RangeWidget::setSingleStep(int step)
{
    rangeSpin->setSingleStep(step);
}

void RangeWidget::rangeChanged()
{
    emit newRange(median->text().toInt() - rangeSpin->value(), median->text().toInt() + rangeSpin->value());
    emit newRange();
}
