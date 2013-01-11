#include <qwidget.h>
#include <qwt_slider.h>

class QLabel;
class QLayout;

class Slider: public QWidget
{
    Q_OBJECT
public:
    Slider(QWidget *parent, int sliderType);

private Q_SLOTS:
    void setNum(double v);

private:
    QwtSlider *createSlider(QWidget *, int sliderType) const;

    QwtSlider *d_slider;
    QLabel *d_label;
};

class SliderDemo : public QWidget
{
public:
    SliderDemo(QWidget *p = NULL);
};
