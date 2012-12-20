#ifndef __PROGRESS_H
#define __PROGRESS_H

#include <QLabel>
#include <QProgressBar>
#include <QStatusBar>
#include <QTimer>
#include <QWidget>
#include <openbr_export.h>

namespace br
{

class BR_EXPORT_GUI Progress : public QStatusBar
{
    Q_OBJECT
    QWidget wSpacer;
    QProgressBar pbProgress;
    QLabel lTimeRemaining;
    QTimer timer;

public:
    explicit Progress(QWidget *parent = 0);

private slots:
    void checkProgress();
};

}

#endif // __PROGRESS_H
