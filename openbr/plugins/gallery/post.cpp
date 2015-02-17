#include <QTcpServer>
#include <QCoreApplication>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

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
//            server->listen(QHostAddress::Any, 8080);
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

} // namespace br

#include "gallery/post.moc"
