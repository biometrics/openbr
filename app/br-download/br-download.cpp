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

static bool processReply(QNetworkReply* reply)
{
    while (!reply->isFinished())
        QCoreApplication::processEvents();
    const QByteArray data = reply->readAll();
    reply->deleteLater();

    if (!permissive && imdecode(Mat(1, data.size(), CV_8UC1, (void*)data.data()), IMREAD_ANYDEPTH | IMREAD_ANYCOLOR).empty())
        return false;

    static QMutex lock;
    QMutexLocker locker(&lock);

    const QByteArray hash = QCryptographicHash::hash(data, QCryptographicHash::Md5);
    br_append_utemplate_contents(stdout, reinterpret_cast<const int8_t*>(hash.data()), reinterpret_cast<const int8_t*>(hash.data()), 3, data.size(), reinterpret_cast<const int8_t*>(data.data()));
    return true;
}

int main(int argc, char *argv[])
{
    QCoreApplication application(argc, argv);

    QNetworkAccessManager networkAccessManager;

    for (int i=1; i<argc; i++) {
        if      (!strcmp(argv[i], "-help"      )) { help(); exit(0); }
        else if (!strcmp(argv[i], "-json"      )) json = true;
        else if (!strcmp(argv[i], "-permissive")) permissive = true;
        else                                      { url_provided = processReply(networkAccessManager.get(QNetworkRequest(QUrl(argv[i])))); }
    }

    if (!url_provided) {
        QFile file;
        file.open(stdin, QFile::ReadOnly);

        while (!file.atEnd()) {
            const QByteArray line = file.readLine();
            const QString url = json ? QJsonDocument::fromJson(line).object().value("URL").toString()
                                     : QString::fromLocal8Bit(line);
            if (!url.isEmpty())
                processReply(networkAccessManager.get(QNetworkRequest(url)));
        }
    }

    return EXIT_SUCCESS;
}
