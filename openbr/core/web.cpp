#include <openbr/openbr_plugin.h>

#include "web.h"

#ifndef BR_EMBEDDED

#include <QCoreApplication>
#include <QTcpServer>
#include <QTcpSocket>

class WebServices : public QTcpServer
{
    Q_OBJECT

public:
    WebServices()
    {
        connect(this, SIGNAL(newConnection()), this, SLOT(handleNewConnection()));
    }

    void listen()
    {
        QTcpServer::listen(QHostAddress::Any, 8080);
    }

private slots:
    void handleNewConnection()
    {
        while (hasPendingConnections()) {
            QTcpSocket *socket = QTcpServer::nextPendingConnection();
            connect(socket, SIGNAL(disconnected()), socket, SLOT(deleteLater()));
            socket->write("HTTP/1.1 200 OK\r\n"
                          "Content-Type: text/html; charset=UTF-8\r\n\r\n"
                          "Hello World!\r\n");
            socket->disconnectFromHost();
        }
    }
};

void br::web()
{
    static WebServices webServices;
    if (webServices.isListening())
        return;

    webServices.listen();
    qDebug("Listening on %s:%d", qPrintable(webServices.serverAddress().toString()), webServices.serverPort());
    while (true)
        QCoreApplication::processEvents();
}

#include "web.moc"

#else // BR_EMBEDDED

void br::web()
{
    qFatal("Web services not supported in embedded builds.");
}

#endif // BR_EMBEDDED
