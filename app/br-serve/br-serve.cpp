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
#include <qhttpserver.h>
#include <qhttprequest.h>
#include <qhttpresponse.h>
#include <openbr/universal_template.h>

static void help()
{
    printf("br-serve [command]\n"
           "==================\n"
           "\n"
           "_br-serve_ converts the command's stdin/stdout into a web service.\n"
           "\n"
           "Optional Arguments\n"
           "------------------\n"
           "* -help       - Print usage information.\n"
           "* -port <int> - The port to communicate on (80 otherwise).");
}

static int port = 80;
static QProcess process;

class Handler : public QObject
{
    Q_OBJECT

public slots:
    void handle(QHttpRequest *request, QHttpResponse *response)
    {
        QByteArray message;

        const QUrlQuery urlQuery(request->url());
        if (urlQuery.hasQueryItem("URL")) {
            process.write(qPrintable(QString(urlQuery.queryItemValue("URL") + "\n")));
            process.waitForReadyRead();
            if (process.error() != QProcess::UnknownError)
                qFatal("%s\n", qPrintable(process.errorString()));
            message = process.readLine();
            response->setHeader("Content-Type", "application/json");
        } else if (urlQuery.hasQueryItem("ImageID")) {
            process.write(qPrintable(QString(urlQuery.queryItemValue("ImageID") + "\n")));
            process.waitForReadyRead();
            if (process.error() != QProcess::UnknownError)
                qFatal("%s\n", qPrintable(process.errorString()));
            response->setHeader("Content-Type", "image/jpeg");
        } else {
            QString path = request->path();
            if (path == "/")
                path = "localhost";
            message = QString("<!DOCTYPE html>\n"
                              "<html>\n"
                              "<head>\n"
                              "  <title>Web Services API</title>\n"
                              "</head>\n"
                              "\n"
                              "<body>\n"
                              "  <h1><a href=\"http://en.wikipedia.org/wiki/Query_string\">Query String</a> Parameters</h1>"
                              "  <ul>\n"
                              "    <li><b>URL</b> - Query URL for image search.</li>\n"
                              "    <li><b>ImageID</b> - Query ImageID for image retrieval.</li>\n"
                              "  </ul>\n"
                              "  <h1>Examples</h1>\n"
                              "  <ul>\n"
                              "    <li>http://%1%2/?URL=data.liblikely.org/misc/lenna.tiff</li>\n"
                              "    <li>http://%1%2/?ImageID=ecaee0b4cd73a76dd2a8060b2909a4a1</li>\n"
                              "  </ul>\n"
                              "</body>\n"
                              "</html>").arg(path, port == 80 ? QString() : (QString(":") + QString::number(port))).toLatin1();
            response->setHeader("Content-Type", "text/html");
        }

        response->setHeader("Content-Length", QString::number(message.size()));
        response->writeHead(200); // everything is OK
        response->write(message);
        response->end();
    }
};

int main(int argc, char *argv[])
{
    QCoreApplication application(argc, argv);

    for (int i=1; i<argc; i++) {
        if      (!strcmp(argv[i], "-help")) { help(); exit(EXIT_SUCCESS); }
        else if (!strcmp(argv[i], "-port")) port = atoi(argv[++i]);
        else                                process.start(argv[i]);
    }

    QHttpServer server;
    Handler handler;
    QObject::connect(&server, SIGNAL(newRequest(QHttpRequest*, QHttpResponse*)),
                     &handler, SLOT(handle(QHttpRequest*, QHttpResponse*)));

    if (!server.listen(port))
        qFatal("Failed to connect to port: %d.", port);

    return application.exec();
}

#include "br-serve.moc"
