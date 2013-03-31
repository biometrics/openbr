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

/*!
 * \ingroup galleries
 * \brief Handle POST requests
 * \author Josh Klontz \cite jklontz
 */
class postGallery : public Gallery
{
    Q_OBJECT

    class TcpServer : public QTcpServer
    {
        QList<qintptr> socketDescriptors;
        QMutex socketDescriptorsLock;

    public:
        QList<qintptr> availableSocketDescriptors()
        {
            QMutexLocker locker(&socketDescriptorsLock);
            QList<qintptr> result(socketDescriptors);
            socketDescriptors.clear();
            return result;
        }

        void addPendingConnection(QTcpSocket *socket)
        {
            QTcpServer::addPendingConnection(socket);
        }

    private:
        void incomingConnection(qintptr socketDescriptor)
        {
            QMutexLocker locker(&socketDescriptorsLock);
            socketDescriptors.append(socketDescriptor);
        }
    };

public:
    static QScopedPointer<TcpServer> server;

    postGallery()
    {
        if (server.isNull()) {
            server.reset(new TcpServer());
            server->listen(QHostAddress::Any, 8080);
            server->moveToThread(QCoreApplication::instance()->thread());
            qDebug("Listening on %s:%d", qPrintable(server->serverAddress().toString()), server->serverPort());
        }
    }

private:
    TemplateList readBlock(bool *done)
    {
        *done = true;
        TemplateList templates;
        foreach (qintptr socketDescriptor, server->availableSocketDescriptors()) {
            File f(".post");
            f.set("socketDescriptor", socketDescriptor);
            templates.append(f);
        }
        if (templates.isEmpty())
            templates.append(File(".null"));
        return templates;
    }

    void write(const Template &t)
    {
        (void) t;
        qFatal("Not supported!");
    }
};

QScopedPointer<postGallery::TcpServer> postGallery::server;

BR_REGISTER(Gallery, postGallery)

/*!
 * \ingroup formats
 * \brief Handle POST requests
 * \author Josh Klontz \cite jklontz
 */
class postFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        Template t(file);
        qDebug() << t.file.flat();
        QTcpSocket *socket = new QTcpSocket();
        socket->setSocketDescriptor(file.get<qintptr>("socketDescriptor"));

        socket->write("HTTP/1.1 200 OK\r\n"
                      "Content-Type: text/html; charset=UTF-8\r\n\r\n"
                      "Hello World!\r\n");
        socket->waitForBytesWritten();

        socket->waitForReadyRead();
        QByteArray data = socket->readAll();
        t.append(imdecode(Mat(1, data.size(), CV_8UC1, data.data()), 1));

        socket->close();
        delete socket;
        return t;
    }

    void write(const Template &t) const
    {
        (void) t;
        qFatal("Not supported!");
    }
};

BR_REGISTER(Format, postFormat)

} // namespace br

#include "qtnetwork.moc"
