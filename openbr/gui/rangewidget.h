#ifndef BR_RANGEWIDGET_H
#define BR_RANGEWIDGET_H

#include <QWidget>
#include <QLineEdit>
#include <QHBoxLayout>
#include <QIntValidator>
#include <QLabel>
#include <QSpinBox>

#include <openbr/openbr_export.h>

namespace br {

class BR_EXPORT RangeWidget : public QWidget
{
    Q_OBJECT

    QVBoxLayout vLayout;
    QHBoxLayout *hLayout;

    QIntValidator *validator;

    QLabel *label;
    QLineEdit *median;
    QSpinBox *rangeSpin;

public:
    explicit RangeWidget(QWidget *parent = 0);

signals:

    void newRange(int, int);
    void newRange();
    
public slots:

    void emitRange() { rangeChanged(); }
    int getLowerBound() const {return median->text().toInt() - rangeSpin->value();}
    int getUpperBound() const {return median->text().toInt() + rangeSpin->value();}
    void rangeChanged();
    void setSingleStep(int step);

};

}

#endif // BR_AGERANGEWIDGET_H
