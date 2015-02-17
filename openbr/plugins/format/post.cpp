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

#include <QTcpSocket>
#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <http_parser.h>

using namespace cv;

namespace br
{

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

        // Read from socket
        QTcpSocket *socket = new QTcpSocket();
        socket->setSocketDescriptor(file.get<qintptr>("socketDescriptor"));
        socket->write("HTTP/1.1 200 OK\r\n"
                      "Content-Type: text/html; charset=UTF-8\r\n\r\n"
                      "Hello World!\r\n");
        socket->waitForBytesWritten();
        socket->waitForReadyRead();
        QByteArray data = socket->readAll();
        socket->close();
        delete socket;

        qDebug() << data;

        // Parse data
        http_parser_settings settings;
        settings.on_body = bodyCallback;
        settings.on_headers_complete = NULL;
        settings.on_header_field = NULL;
        settings.on_header_value = NULL;
        settings.on_message_begin = NULL;
        settings.on_message_complete = NULL;
        settings.on_status_complete = NULL;
        settings.on_url = NULL;

        {
            QByteArray body;
            http_parser parser;
            http_parser_init(&parser, HTTP_REQUEST);
            parser.data = &body;
            http_parser_execute(&parser, &settings, data.data(), data.size());
            data = body;
        }

        data.prepend("HTTP/1.1 200 OK");
        QByteArray body;
        { // Image data is two layers deep
            http_parser parser;
            http_parser_init(&parser, HTTP_BOTH);
            parser.data = &body;
            http_parser_execute(&parser, &settings, data.data(), data.size());
        }

        t.append(imdecode(Mat(1, body.size(), CV_8UC1, body.data()), 1));
        return t;
    }

    void write(const Template &t) const
    {
        (void) t;
        qFatal("Not supported!");
    }

    static int bodyCallback(http_parser *parser, const char *at, size_t length)
    {
        QByteArray *byteArray = (QByteArray*)parser->data;
        *byteArray = QByteArray(at, length);
        return 0;
    }
};

BR_REGISTER(Format, postFormat)

} // namespace br

#include "format/post.moc"
