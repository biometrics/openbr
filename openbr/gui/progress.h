#ifndef BR_PROGRESS_H
#define BR_PROGRESS_H

#include <QLabel>
#include <QProgressBar>
#include <QStatusBar>
#include <QTimer>
#include <QWidget>
#include <openbr/openbr_export.h>

namespace br
{

class BR_EXPORT Progress : public QStatusBar
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

#endif // BR_PROGRESS_H
