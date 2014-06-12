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
#include <qhttpserver.h>
#include <qhttprequest.h>
#include <qhttpresponse.h>
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

QProcess process;

class Handler : public QObject
{
    Q_OBJECT

public slots:
    void handle(QHttpRequest *request, QHttpResponse *response)
    {
        (void) request;
        response->setHeader("Content-Length", QString::number(11));
        response->writeHead(200); // everything is OK
        response->write("Hello World");
        response->end();
    }
};

int main(int argc, char *argv[])
{
    QCoreApplication application(argc, argv);
    int port = 80;

    for (int i=1; i<argc; i++) {
        if      (!strcmp(argv[i], "-help")) { help(); exit(EXIT_SUCCESS); }
        else if (!strcmp(argv[i], "-port")) port = atoi(argv[++i]);
        else                                process.execute(argv[i]);
    }

    QHttpServer server;
    Handler handler;
    QObject::connect(&server, SIGNAL(newRequest(QHttpRequest*, QHttpResponse*)),
                     &handler, SLOT(handle(QHttpRequest*, QHttpResponse*)));

    server.listen(port);
    return application.exec();
}

#include "br-serve.moc"
