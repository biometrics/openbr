#include "wheelbox.h"
#include <qwt_wheel.h>
#include <qlcdnumber.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qevent.h>
#include <qapplication.h>

class Wheel: public QwtWheel
{
public:
    Wheel(WheelBox *parent):
        QwtWheel(parent)
    {
        setFocusPolicy(Qt::WheelFocus);
        parent->installEventFilter(this);
    }

    virtual bool eventFilter(QObject *object, QEvent *ev)
    {
        if ( ev->type() == QEvent::Wheel )
        {
            QWheelEvent *we = (QWheelEvent *)ev;

            QWheelEvent wheelEvent(QPoint(5, 5), we->delta(), 
                we->buttons(), we->modifiers(),
                we->orientation());
                
            QApplication::sendEvent(this, &wheelEvent);
            return true;
        }
        return QwtWheel::eventFilter(object, ev);
    }
};

WheelBox::WheelBox(const QString &title,
        double min, double max, double stepSize, QWidget *parent):
    QWidget(parent)
{

    d_number = new QLCDNumber(this);
    d_number->setSegmentStyle(QLCDNumber::Filled);
    d_number->setAutoFillBackground(true);
    d_number->setFixedHeight(d_number->sizeHint().height() * 2 );
    d_number->setFocusPolicy(Qt::WheelFocus);

    QPalette pal(Qt::black);
    pal.setColor(QPalette::WindowText, Qt::green);
    d_number->setPalette(pal);
    
    d_wheel = new Wheel(this);
    d_wheel->setOrientation(Qt::Vertical);
    d_wheel->setRange(min, max, stepSize);
    d_wheel->setFixedSize(
        qRound(d_number->height() / 2.5), d_number->height());

    d_number->setFocusProxy(d_wheel);

    QFont font("Helvetica", 10);
    font.setBold(true);

    d_label = new QLabel(title, this);
    d_label->setFont(font);

    QHBoxLayout *hLayout = new QHBoxLayout;
    hLayout->setContentsMargins(0, 0, 0, 0);
    hLayout->setSpacing(2);
    hLayout->addWidget(d_number, 10);
    hLayout->addWidget(d_wheel);

    QVBoxLayout *vLayout = new QVBoxLayout(this);
    vLayout->addLayout(hLayout, 10);
    vLayout->addWidget(d_label, 0, Qt::AlignTop | Qt::AlignHCenter);

    connect(d_wheel, SIGNAL(valueChanged(double)), 
        d_number, SLOT(display(double)));
    connect(d_wheel, SIGNAL(valueChanged(double)), 
        this, SIGNAL(valueChanged(double)));
}

void WheelBox::setValue(double value)
{
    d_wheel->setValue(value);
    d_number->display(value);
}

double WheelBox::value() const
{
    return d_wheel->value();
}
