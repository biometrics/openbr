#include <qapplication.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qwt_slider.h>
#include <qwt_scale_engine.h>
#include <qwt_scale_map.h>
#include "sliders.h"

class Layout: public QBoxLayout
{
public:
    Layout(Qt::Orientation o, QWidget *parent = NULL):
        QBoxLayout(QBoxLayout::LeftToRight, parent)
    {
        if ( o == Qt::Vertical )
            setDirection(QBoxLayout::TopToBottom);

        setSpacing(20);
        setMargin(0);
    }
};

Slider::Slider(QWidget *parent, int sliderType):
    QWidget(parent)
{
    d_slider = createSlider(this, sliderType);

    QFlags<Qt::AlignmentFlag> alignment;
    switch(d_slider->scalePosition())
    {
        case QwtSlider::NoScale:
            if ( d_slider->orientation() == Qt::Horizontal )
                alignment = Qt::AlignHCenter | Qt::AlignTop;
            else
                alignment = Qt::AlignVCenter | Qt::AlignLeft;
            break;
        case QwtSlider::LeftScale:
            alignment = Qt::AlignVCenter | Qt::AlignRight;
            break;
        case QwtSlider::RightScale:
            alignment = Qt::AlignVCenter | Qt::AlignLeft;
            break;
        case QwtSlider::TopScale:
            alignment = Qt::AlignHCenter | Qt::AlignBottom;
            break;
        case QwtSlider::BottomScale:
            alignment = Qt::AlignHCenter | Qt::AlignTop;
            break;
    }

    d_label = new QLabel("0", this);
    d_label->setAlignment(alignment);
    d_label->setFixedWidth(d_label->fontMetrics().width("10000.9"));

    connect(d_slider, SIGNAL(valueChanged(double)), SLOT(setNum(double)));

    QBoxLayout *layout;
    if ( d_slider->orientation() == Qt::Horizontal )
        layout = new QHBoxLayout(this);
    else
        layout = new QVBoxLayout(this);

    layout->addWidget(d_slider);
    layout->addWidget(d_label);
}

QwtSlider *Slider::createSlider(QWidget *parent, int sliderType) const
{
    QwtSlider *slider = NULL;

    switch( sliderType )
    {
        case 0:
        {
            slider = new QwtSlider(parent, Qt::Horizontal, 
                QwtSlider::TopScale, QwtSlider::Trough);
            slider->setHandleSize( 30, 16 );
            slider->setRange(-10.0, 10.0, 1.0, 0); // paging disabled
            break;
        }
        case 1:
        {
            slider = new QwtSlider(parent, Qt::Horizontal, 
                QwtSlider::NoScale, QwtSlider::Trough | QwtSlider::Groove );
            slider->setRange(0.0, 1.0, 0.01, 5);
            break;
        }
        case 2:
        {
            slider = new QwtSlider(parent, Qt::Horizontal, 
                QwtSlider::BottomScale, QwtSlider::Groove);
            slider->setHandleSize( 12, 25 );
            slider->setRange(1000.0, 3000.0, 10.0, 10);
            break;
        }
        case 3:
        {
            slider = new QwtSlider(parent, Qt::Vertical, 
                QwtSlider::LeftScale, QwtSlider::Groove);
            slider->setRange(0.0, 100.0, 1.0, 5);
            slider->setScaleMaxMinor(5);
            break;
        }
        case 4:
        {
            slider = new QwtSlider(parent, Qt::Vertical, 
                QwtSlider::NoScale, QwtSlider::Trough);
            slider->setRange(0.0,100.0,1.0, 10);
            break;
        }
        case 5:
        {
            slider = new QwtSlider(parent, Qt::Vertical, 
                QwtSlider::RightScale, QwtSlider::Trough | QwtSlider::Groove);
            slider->setScaleEngine(new QwtLog10ScaleEngine);
            slider->setHandleSize( 20, 32 );
            slider->setBorderWidth(1);
            slider->setRange(0.0, 4.0, 0.01);
            slider->setScale(1.0, 1.0e4);
            slider->setScaleMaxMinor(10);
            break;
        }
    }

    if ( slider )
    {
        QString name( "Slider %1" );
        slider->setObjectName( name.arg( sliderType ) );
    }

    return slider;
}

void Slider::setNum( double v )
{
    if ( d_slider->scaleMap().transformation()->type() ==
        QwtScaleTransformation::Log10 )
    {
        v = qPow(10.0, v);
    }

    QString text;
    text.setNum(v, 'f', 1);

    d_label->setText(text);
}

SliderDemo::SliderDemo(QWidget *p): 
    QWidget(p)
{
    int i;

    Layout *hSliderLayout = new Layout(Qt::Vertical);
    for ( i = 0; i < 3; i++ )
        hSliderLayout->addWidget(new Slider(this, i));
    hSliderLayout->addStretch();

    Layout *vSliderLayout = new Layout(Qt::Horizontal);
    for ( ; i < 6; i++ )
        vSliderLayout->addWidget(new Slider(this, i));

    QLabel *vTitle = new QLabel("Vertical Sliders", this);
    vTitle->setFont(QFont("Helvetica", 14, QFont::Bold));
    vTitle->setAlignment(Qt::AlignHCenter);

    Layout *layout1 = new Layout(Qt::Vertical);
    layout1->addWidget(vTitle, 0);
    layout1->addLayout(vSliderLayout, 10);

    QLabel *hTitle = new QLabel("Horizontal Sliders", this);
    hTitle->setFont(vTitle->font());
    hTitle->setAlignment(Qt::AlignHCenter);

    Layout *layout2 = new Layout(Qt::Vertical);
    layout2->addWidget(hTitle, 0);
    layout2->addLayout(hSliderLayout, 10);

    Layout *mainLayout = new Layout(Qt::Horizontal, this);
    mainLayout->addLayout(layout1);
    mainLayout->addLayout(layout2, 10);
}

int main (int argc, char **argv)
{
    QApplication a(argc, argv);

    QApplication::setFont(QFont("Helvetica",10));

    SliderDemo w;
    w.show();

    return a.exec();
}
