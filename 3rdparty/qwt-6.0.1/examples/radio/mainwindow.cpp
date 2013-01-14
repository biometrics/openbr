#include <qlayout.h>
#include "tunerfrm.h"
#include "ampfrm.h"
#include "mainwindow.h"

MainWindow::MainWindow(): 
    QWidget()
{
    TunerFrame *frmTuner = new TunerFrame(this);
    frmTuner->setFrameStyle(QFrame::Panel|QFrame::Raised);

    AmpFrame *frmAmp = new AmpFrame(this);
    frmAmp->setFrameStyle(QFrame::Panel|QFrame::Raised);

    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->setMargin(0);
    layout->setSpacing(0);
    layout->addWidget(frmTuner);
    layout->addWidget(frmAmp);
    
    connect(frmTuner, SIGNAL(fieldChanged(double)), 
        frmAmp, SLOT(setMaster(double)));

    frmTuner->setFreq(90.0);    

    setPalette( QPalette( QColor( 192, 192, 192 ) ) );
    updateGradient();
}

void MainWindow::resizeEvent( QResizeEvent * )
{
#ifdef Q_WS_X11
    updateGradient();
#endif
}

void MainWindow::updateGradient()
{
    QPalette pal = palette();

    const QColor buttonColor = pal.color( QPalette::Button );
    const QColor lightColor = pal.color( QPalette::Light );
    const QColor midLightColor = pal.color( QPalette::Midlight );

#ifdef Q_WS_X11
    // Qt 4.7.1: QGradient::StretchToDeviceMode is buggy on X11

    QLinearGradient gradient( rect().topLeft(), rect().topRight() );
#else
    QLinearGradient gradient( 0, 0, 1, 0 );
    gradient.setCoordinateMode( QGradient::StretchToDeviceMode );
#endif

    gradient.setColorAt( 0.0, midLightColor );
    gradient.setColorAt( 0.7, buttonColor );
    gradient.setColorAt( 1.0, buttonColor );

    pal.setBrush( QPalette::Window, gradient );
    setPalette( pal );
}
