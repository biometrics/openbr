#include <QDesktopServices>
#include <QFile>
#include <QMessageBox>
#include <QString>
#include <QUrl>
#include <openbr/openbr.h>

#include "help.h"

/**** HELP ****/
/*** PUBLIC ***/
br::Help::Help(QWidget *parent)
    : QMenu(parent)
{
    actions.append(QSharedPointer<QAction>(addAction("About")));
    connect(actions.last().data(), SIGNAL(triggered()), this, SLOT(showAbout()));
    actions.append(QSharedPointer<QAction>(addAction("Documentation")));
    connect(actions.last().data(), SIGNAL(triggered()), this, SLOT(showDocumentation()));
    actions.append(QSharedPointer<QAction>(addAction("License")));
    connect(actions.last().data(), SIGNAL(triggered()), this, SLOT(showLicense()));
}

/*** PUBLIC SLOTS ***/
void br::Help::showAbout()
{
    QMessageBox::about(this, "About", br_about());
}

void br::Help::showDocumentation()
{
    QDesktopServices::openUrl(QUrl(QString("file:///%1/doc/html/index.html").arg(br_sdk_path())));
}

void br::Help::showLicense()
{
    QFile file(QString("%1/LICENSE.txt").arg(br_sdk_path()));
    file.open(QFile::ReadOnly);
    QString license = file.readAll();
    file.close();

    QMessageBox::about(this, "License", license);
}

#include "moc_help.cpp"
