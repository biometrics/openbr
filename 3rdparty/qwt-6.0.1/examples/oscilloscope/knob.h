#ifndef _KNOB_H_
#define _KNOB_H_

#include <qwidget.h>

class QwtKnob;
class QLabel;

class Knob: public QWidget
{
    Q_OBJECT

public:
    Knob(const QString &title, 
        double min, double max, QWidget *parent = NULL);

    virtual QSize sizeHint() const;

    void setValue(double value);
    double value() const;

Q_SIGNALS:
    double valueChanged(double);

protected:
    virtual void resizeEvent(QResizeEvent *);

private:
    QwtKnob *d_knob;
    QLabel *d_label;
};

#endif
