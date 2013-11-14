#include <openbr/openbr.h>
#include <QDebug>

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
    timer.start(5000);
}

/*** PRIVATE SLOTS ***/
void br::Progress::checkProgress()
{
    const int progress = 100 * br_progress();
    const bool visible = progress >= 0;

    if (visible) {
        showMessage(br_most_recent_message());
        pbProgress.setValue(progress);
        if (progress > 100) pbProgress.setMaximum(0);
        else                pbProgress.setMaximum(100);
    } else {
        clearMessage();
    }

    pbProgress.setVisible(visible);
}

#include "moc_progress.cpp"
