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
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <cstring>
#include <string>
#include <openbr/universal_template.h>

using namespace cv;
using namespace std;

static void help()
{
    printf("br-download [URL] [args]\n"
           "========================\n"
           "* __stdin__  - URLs/JSON\n"
           "* __stdout__ - Templates (raw data)\n"
           "\n"
           "_br-download_ retrieves and verifies the contents of an image URL.\n"
           "If no URL is provided, download reads newline-separated image URLs from _stdin_.\n"
           "Download writes templates containing the contents of the URL to _stdout_.\n"
           "\n"
           "Download is expected to filter out URLs that are not valid images.\n"
           "Download may do this by checking the file header or decoding the file.\n"
           "\n"
           "Optional Arguments\n"
           "------------------\n"
           "* -help - Print usage information.\n"
           "* -json - Input JSON instead of URLs.\n"
           "* -permissive - Do not attempt to verify the contents of an image URL (verify otherwise).\n");
}

static bool json = false;
static bool permissive = false;
static bool url_provided = false;

static void process(QString url, const QByteArray &metadata, QNetworkAccessManager &nam)
{
    url = url.simplified();
    if (url.isEmpty())
        return;

    if (url.startsWith("file://"))
        url = url.mid(7);

    QIODevice *device = NULL;
    if (QFileInfo(url).exists()) {
        device = new QFile(url);
        device->open(QIODevice::ReadOnly);
    } else {
        QNetworkReply *reply = nam.get(QNetworkRequest(url));
        while (!reply->isFinished())
            QCoreApplication::processEvents();

        if (reply->error() != QNetworkReply::NoError) {
            qDebug() << reply->errorString() << url;
            reply->deleteLater();
        } else {
            device = reply;
        }
    }

    if (!device)
        return;

    const QByteArray data = device->readAll();
    delete device;
    device = NULL;

    if (!permissive && imdecode(Mat(1, data.size(), CV_8UC1, (void*)data.data()), IMREAD_ANYDEPTH | IMREAD_ANYCOLOR).empty())
        return;

    const QByteArray hash = QCryptographicHash::hash(data, QCryptographicHash::Md5);
    br_append_utemplate_contents(stdout, reinterpret_cast<const unsigned char*>(hash.data()), reinterpret_cast<const unsigned char*>(hash.data()), 3, data.size(), reinterpret_cast<const unsigned char*>(data.data()));

    if (!metadata.isEmpty()) {
        const QByteArray metadataHash = QCryptographicHash::hash(metadata, QCryptographicHash::Md5);
        br_append_utemplate_contents(stdout, reinterpret_cast<const unsigned char*>(hash.data()), reinterpret_cast<const unsigned char*>(metadataHash.data()), 2, metadata.size() + 1 /* include null terminator */, reinterpret_cast<const unsigned char*>(metadata.data()));
    }
}

int main(int argc, char *argv[])
{
    QCoreApplication application(argc, argv);
    QNetworkAccessManager nam;

    for (int i=1; i<argc; i++) {
        if      (!strcmp(argv[i], "-help"      )) { help(); exit(EXIT_SUCCESS); }
        else if (!strcmp(argv[i], "-json"      )) json = true;
        else if (!strcmp(argv[i], "-permissive")) permissive = true;
        else                                      { url_provided = true; process(argv[i], QByteArray(), nam); }
    }

    if (!url_provided) {
        QFile file;
        file.open(stdin, QFile::ReadOnly);
        while (!file.atEnd()) {
            const QByteArray line = file.readLine().simplified();
            if (line.isEmpty())
                continue;

            QJsonParseError error;
            process(json ? QJsonDocument::fromJson(line, &error).object().value("URL").toString()
                         : QString::fromLatin1(line),
                    json ? line : QByteArray(),
                    nam);

            if (json && (error.error != QJsonParseError::NoError))
                qDebug() << error.errorString();
        }
    }

    return EXIT_SUCCESS;
}
