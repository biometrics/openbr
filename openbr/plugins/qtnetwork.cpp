#include <QCoreApplication>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QTcpServer>
#include <QTcpSocket>
#include <opencv2/highgui/highgui.hpp>
#include <openbr/openbr_plugin.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup formats
 * \brief Reads image files from the web.
 * \author Josh Klontz \cite jklontz
 */
class urlFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        Template t;

        QNetworkAccessManager networkAccessManager;
        QNetworkRequest request(QString(file.name).remove(".url"));
        request.setAttribute(QNetworkRequest::CacheLoadControlAttribute, QNetworkRequest::AlwaysNetwork);
        QNetworkReply *reply = networkAccessManager.get(request);

        while (!reply->isFinished()) QCoreApplication::processEvents();
        if (reply->error()) qWarning("%s (%s)", qPrintable(reply->errorString()), qPrintable(QString::number(reply->error())));

        QByteArray data = reply->readAll();
        delete reply;

        Mat m = imdecode(Mat(1, data.size(), CV_8UC1, data.data()), 1);
        if (m.data) t.append(m);

        return t;
    }

    void write(const Template &t) const
    {
        (void) t;
        qFatal("Not supported.");
    }
};

BR_REGISTER(Format, urlFormat)

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
            connect(socket, SIGNAL(readyRead()), this, SLOT(read()));
//            socket->write("HTTP/1.1 200 OK\r\n"
//                          "Content-Type: text/html; charset=UTF-8\r\n\r\n"
//                          "Hello World!\r\n");
//            socket->disconnectFromHost();
        }
    }

    void read()
    {
        QTcpSocket *socket = dynamic_cast<QTcpSocket*>(QObject::sender());
        if (socket == NULL) return;
        QByteArray data = socket->readAll();
        qDebug() << data;
    }
};

//static void web()
//{
//    static WebServices webServices;
//    if (webServices.isListening())
//        return;

//    webServices.listen();
//    qDebug("Listening on %s:%d", qPrintable(webServices.serverAddress().toString()), webServices.serverPort());
//    while (true)
//        QCoreApplication::processEvents();
//}

} // namespace br

#include "qtnetwork.moc"
