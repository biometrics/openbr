#include <qlayout.h>
#include <qlabel.h>
#include <qwt_wheel.h>
#include <qwt_slider.h>
#include <qwt_thermo.h>
#include <qwt_math.h>
#include "tunerfrm.h"

#if QT_VERSION < 0x040600
#define qFastSin(x) ::sin(x)
#define qFastCos(x) ::cos(x)
#endif

class TuningThermo: public QWidget
{
public:
    TuningThermo(QWidget *parent):
        QWidget(parent)
    {
        d_thermo = new QwtThermo(this);
        d_thermo->setOrientation(Qt::Horizontal, QwtThermo::NoScale);
        d_thermo->setRange(0.0, 1.0);
        d_thermo->setFillBrush( Qt::green );

        QLabel *label = new QLabel("Tuning", this);
        label->setAlignment(Qt::AlignCenter);

        QVBoxLayout *layout = new QVBoxLayout(this);
        layout->setMargin(0);
        layout->addWidget(d_thermo);
        layout->addWidget(label);

        setFixedWidth(3 * label->sizeHint().width());
    }

    void setValue(double value)
    {
        d_thermo->setValue(value);
    }

private:
    QwtThermo *d_thermo;
};

TunerFrame::TunerFrame(QWidget *parent): 
    QFrame(parent)
{
    d_sliderFrequency = new QwtSlider(this, Qt::Horizontal, QwtSlider::TopScale);
    d_sliderFrequency->setRange(87.5, 108, 0.01, 10);
    d_sliderFrequency->setScaleMaxMinor(5);
    d_sliderFrequency->setScaleMaxMajor(12);
    d_sliderFrequency->setHandleSize(80, 20 );
    d_sliderFrequency->setBorderWidth(1);

    d_thermoTune = new TuningThermo(this);

    d_wheelFrequency = new QwtWheel(this);
    d_wheelFrequency->setMass(0.5);
    d_wheelFrequency->setRange(87.5, 108, 0.01);
    d_wheelFrequency->setTotalAngle(3600.0);
    d_wheelFrequency->setFixedHeight(30);


    connect(d_wheelFrequency, SIGNAL(valueChanged(double)), SLOT(adjustFreq(double)));
    connect(d_sliderFrequency, SIGNAL(valueChanged(double)), SLOT(adjustFreq(double)));

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setMargin(10);
    mainLayout->setSpacing(5);
    mainLayout->addWidget(d_sliderFrequency);

    QHBoxLayout *hLayout = new QHBoxLayout;
    hLayout->setMargin(0);
    hLayout->addWidget(d_thermoTune, 0);
    hLayout->addStretch(5);
    hLayout->addWidget(d_wheelFrequency, 2);

    mainLayout->addLayout(hLayout);
}

void TunerFrame::adjustFreq(double frq)
{
    const double factor = 13.0 / (108 - 87.5);

    const double x = (frq - 87.5) * factor;
    const double field = qwtSqr(qFastSin(x) * qFastCos(4.0 * x));
    
    d_thermoTune->setValue(field);  

    if (d_sliderFrequency->value() != frq) 
        d_sliderFrequency->setValue(frq);
    if (d_wheelFrequency->value() != frq) 
        d_wheelFrequency->setValue(frq);

    Q_EMIT fieldChanged(field);   
}

void TunerFrame::setFreq(double frq)
{
    d_wheelFrequency->setValue(frq);
}
