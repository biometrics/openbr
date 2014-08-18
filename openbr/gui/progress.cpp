#include <openbr/openbr.h>
#include <openbr/openbr_plugin.h>

#include "progress.h"

/**** PROGRESS ****/
/*** PUBLIC ***/
br::Progress::Progress(QWidget *parent)
    : QStatusBar(parent)
{
    setContentsMargins(0, 0, 0, 0);
    pbProgress.setVisible(false);
    pbProgress.setMaximum(100);
    pbProgress.setMinimum(0);
    pbProgress.setValue(0);
    pbProgress.setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
    pbProgress.setTextVisible(false);

    addWidget(&wSpacer, 1);
    addPermanentWidget(&pbProgress);
    addPermanentWidget(&lTimeRemaining);
    connect(&timer, SIGNAL(timeout()), this, SLOT(checkProgress()));
    timer.start(1000);
}

/*** PRIVATE SLOTS ***/
void br::Progress::checkProgress()
{
    const int progress = 100 * br_progress();
    const bool visible = progress > 0 && progress < 100;

    if (visible) {
        showMessage(Globals->mostRecentMessage);
        pbProgress.setValue(progress);
        if (progress > 100) pbProgress.setMaximum(0);
        else                pbProgress.setMaximum(100);

        int s = br_time_remaining();
        if (s >= 0) {
            int h = s / (60*60);
            int m = (s - h*60*60) / 60;
            s = (s - h*60*60 - m*60);
            lTimeRemaining.setText(QString("%1:%2:%3").arg(h, 2, 10, QLatin1Char('0')).arg(m, 2, 10, QLatin1Char('0')).arg(s, 2, 10, QLatin1Char('0')));
        } else {
            lTimeRemaining.clear();
        }
    } else {
        clearMessage();
        lTimeRemaining.clear();
    }

    pbProgress.setVisible(visible);
}

#include "moc_progress.cpp"
