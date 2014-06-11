/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2014 Noblis                                                     *
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

#include <QtCore>
#include <QtNetwork>
#include <cstdio>
#include <cstring>
#include <string>
#include <openbr/universal_template.h>

static void help()
{
    printf("br-serve [command]\n"
           "===================\n"
           "\n"
           "_br-serve_ converts the command's stdin/stdout into a webservice.\n"
           "\n"
           "Optional Arguments\n"
           "------------------\n"
           "* -help       - Print usage information.\n"
           "* -port <int> - The port to communicate on (80 otherwise).");
}

static int port = 80;

class Server : public QObject
{
    Q_OBJECT
    QTcpServer *tcpServer;

public:
    Server()
        : tcpServer(new QTcpServer(this))
    {
        if (!tcpServer->listen(QHostAddress::Any, port)) {
            qDebug() << tcpServer->errorString();
            exit(EXIT_FAILURE);
            return;
        }

        connect(tcpServer, SIGNAL(newConnection()), this, SLOT(newConnection()));
    }

private slots:
    void newConnection()
    {
        QByteArray block = "HTTP/1.0 200 Ok\r\n"
                           "Content-Type: text/html; charset=\"utf-8\"\r\n"
                           "\r\n"
                           "<h1>Hello World!</h1>\n";

        QTcpSocket *clientConnection = tcpServer->nextPendingConnection();
        connect(clientConnection, SIGNAL(disconnected()),
                clientConnection, SLOT(deleteLater()));

        clientConnection->write(block);
        clientConnection->disconnectFromHost();
    }
};

int main(int argc, char *argv[])
{
    QCoreApplication application(argc, argv);

    for (int i=1; i<argc; i++) {
        if      (!strcmp(argv[i], "-help")) { help(); exit(EXIT_SUCCESS); }
        else if (!strcmp(argv[i], "-port")) { port = atoi(argv[++i]); }
    }

    Server server;
    return application.exec();
}

#include "br-serve.moc"
