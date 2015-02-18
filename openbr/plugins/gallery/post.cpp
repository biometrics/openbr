/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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
