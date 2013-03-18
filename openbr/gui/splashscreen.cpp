#include <QPixmap>
#include <openbr/openbr_plugin.h>

#include "splashscreen.h"

using namespace br;

/**** SPLASH_SCREEN ****/
/*** PUBLIC ***/
SplashScreen::SplashScreen()
    : QSplashScreen(QPixmap(":/icons/mm.png").scaledToWidth(384, Qt::SmoothTransformation))
{
    connect(&timer, SIGNAL(timeout()), this, SLOT(updateMessage()));
    timer.start(100);
}

/*** PROTECTED ***/
void SplashScreen::closeEvent(QCloseEvent *event)
{
    QSplashScreen::closeEvent(event);
    event->accept();
    timer.stop();
}

/*** PRIVATE SLOTS ***/
void SplashScreen::updateMessage()
{
    showMessage("Version " + Context::version() + " " + QChar(169) + " MITRE 2012\n" + Globals->mostRecentMessage, Qt::AlignHCenter | Qt::AlignBottom);
}

#include "moc_splashscreen.cpp"
