#include <qapplication.h>
#include <qpainter.h>
#include <qwt_math.h>
#include <qwt_symbol.h>
#include <qwt_curve_fitter.h>
#include "curvdemo2.h"


//------------------------------------------------------------
//      curvdemo2
//
//  This example shows a simple animation featuring
//  with several QwtPlotCurves
//
//------------------------------------------------------------

//
//   Array Sizes
//
const int Size = 15;
const int USize = 13;

//
//   Arrays holding the values
//
double xval[Size];
double yval[Size];
double zval[Size];
double uval[USize];
double vval[USize];


//
//  CONSTRUCT MAIN WINDOW
//
MainWin::MainWin(): 
    QFrame()
{
    setFrameStyle(QFrame::Box|QFrame::Raised);
    setLineWidth(2);
    setMidLineWidth(3);
    
    const QColor bgColor(30,30,50);

    QPalette p = palette();
    p.setColor(backgroundRole(), bgColor);
    setPalette(p);

    QwtSplineCurveFitter* curveFitter; 

    //
    //  curve 1
    // 
    int i = 0;
    xMap[i].setScaleInterval(-1.5, 1.5);
    yMap[i].setScaleInterval(0.0, 6.28);

    curve[i].setPen(QPen(QColor(150,150,200),2));
    curve[i].setStyle(QwtPlotCurve::Lines);
    curve[i].setCurveAttribute(QwtPlotCurve::Fitted, true);
    curveFitter = new QwtSplineCurveFitter();
    curveFitter->setSplineSize(150);
    curve[i].setCurveFitter(curveFitter);

    QwtSymbol *symbol = new QwtSymbol(QwtSymbol::XCross);
    symbol->setPen(QPen(Qt::yellow,2));
    symbol->setSize(7);
    
    curve[i].setSymbol(symbol);

    curve[i].setRawSamples(yval,xval,Size);
    
    //
    // curve 2
    //
    i++;
    xMap[i].setScaleInterval(0.0, 6.28);
    yMap[i].setScaleInterval(-3.0, 1.1);
    curve[i].setPen(QPen(QColor(200,150,50)));
    curve[i].setStyle(QwtPlotCurve::Sticks);
    curve[i].setSymbol(new QwtSymbol(QwtSymbol::Ellipse,
        QColor(Qt::blue), QColor(Qt::yellow), QSize(5,5)));

    curve[i].setRawSamples(xval,zval,Size);

    
    //
    //  curve 3
    // 
    i++;
    xMap[i].setScaleInterval(-1.1, 3.0);
    yMap[i].setScaleInterval(-1.1, 3.0);
    curve[i].setStyle(QwtPlotCurve::Lines);
    curve[i].setCurveAttribute(QwtPlotCurve::Fitted, true);
    curve[i].setPen(QColor(100,200,150));
    curveFitter = new QwtSplineCurveFitter();
    curveFitter->setFitMode(QwtSplineCurveFitter::ParametricSpline);
    curveFitter->setSplineSize(200);
    curve[i].setCurveFitter(curveFitter);

    curve[i].setRawSamples(yval,zval,Size);


    //
    //  curve 4
    //
    i++;
    xMap[i].setScaleInterval(-5, 1.1);
    yMap[i].setScaleInterval(-1.1, 5.0);
    curve[i].setStyle(QwtPlotCurve::Lines);
    curve[i].setCurveAttribute(QwtPlotCurve::Fitted, true);
    curve[i].setPen(QColor(Qt::red));
    curveFitter = new QwtSplineCurveFitter();
    curveFitter->setSplineSize(200);
    curve[i].setCurveFitter(curveFitter);

    curve[i].setRawSamples(uval,vval,USize);

    //
    //  initialize values
    //
    double base = 2.0 * M_PI / double(USize - 1);
    double toggle = 1.0; 
    for (i = 0; i < USize; i++)
    {
        uval[i] =  toggle * qCos( double(i) * base);
        vval[i] =  toggle * qSin( double(i) * base);
            
        if (toggle == 1.0)
           toggle = 0.5;
        else
           toggle = 1.0;
    }

    newValues();

    //
    // start timer
    //
    (void)startTimer(250);  
}

void MainWin::paintEvent(QPaintEvent *event)
{
    QFrame::paintEvent(event);

    QPainter painter(this);
    painter.setClipRect(contentsRect());
    drawContents(&painter);
}

void MainWin::drawContents(QPainter *painter)
{
    const QRect &r = contentsRect();

    for ( int i = 0; i < curveCount; i++ )
    {
        xMap[i].setPaintInterval(r.left(), r.right());
        yMap[i].setPaintInterval(r.top(), r.bottom());
        curve[i].draw(painter, xMap[i], yMap[i], r);
    }
}

//
//  TIMER EVENT
//
void MainWin::timerEvent(QTimerEvent *)
{
    newValues();
    repaint();
}

//
//  RE-CALCULATE VALUES
//
void MainWin::newValues()
{
    int i;
    static double phs = 0.0;
    double s,c,u;
    
    for (i=0;i<Size;i++)
    {
        xval[i] = 6.28 * double(i) / double(Size -1);
        yval[i] = qSin(xval[i] - phs);
        zval[i] = qCos(3.0 * (xval[i] + phs));
    }
    
    s = 0.25 * qSin(phs);
    c = qSqrt(1.0 - s*s);
    for (i=0; i<USize;i++)
    {
        u = uval[i];
        uval[i] = uval[i] * c - vval[i] * s;
        vval[i] = vval[i] * c + u * s;
    }
    
    phs += 0.0628;
    if (phs > 6.28)
       phs = 0.0;
    
}

int main (int argc, char **argv)
{
    QApplication a(argc, argv);

    MainWin w;

    w.resize(300,300);
    w.show();

    return a.exec();
}
