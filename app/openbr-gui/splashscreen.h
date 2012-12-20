#ifndef __SPLASHSCREEN_H
#define __SPLASHSCREEN_H

#include <QCloseEvent>
#include <QLabel>
#include <QSplashScreen>
#include <QTimer>
#include <QWidget>
#include <openbr_export.h>

namespace br
{

class BR_EXPORT_GUI SplashScreen : public QSplashScreen
{
    Q_OBJECT
    QTimer timer;

public:
    SplashScreen();

protected:
    void closeEvent(QCloseEvent *event);

private slots:
    void updateMessage();
};

} // namespace br

#endif // __SPLASHSCREEN_H
