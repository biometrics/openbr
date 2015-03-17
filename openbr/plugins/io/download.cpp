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

#include <QtNetwork>
#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Downloads an image from a URL
 * \author Josh Klontz \cite jklontz
 */
class DownloadTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_ENUMS(Mode)
    Q_PROPERTY(Mode mode READ get_mode WRITE set_mode RESET reset_mode STORED false)

public:
    enum Mode { Permissive,
                Encoded,
                Decoded };
private:
    BR_PROPERTY(Mode, mode, Encoded)

    // The reasons for this data structure are as follows:
    // 1) The QNetworkAccessManager must be used in the thread that _created_ it,
    //    hence the use of `QThreadStorage`.
    // 2) The QThreadStorage must be deleted _after_ the threads that added QNetworkAccessManager
    //    to it are deleted, hence the `static` ensuring that `nam` is deleted at program termination,
    //    long after the threads that created QNetworkAccessManager are deleted.
    static QThreadStorage<QNetworkAccessManager*> nam;

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        QString url = src.file.get<QString>("URL", src.file.name).simplified();
        if (!url.contains("://"))
            url = "file://" + url;
        dst.file.set("URL", url);

        static const QRegularExpression regExp("file:///[A-Z]:/");

        if (url.contains(regExp))
            url = url.mid(8);
        else if (url.startsWith("file://"))
            url = url.mid(7);

        QIODevice *device = NULL;
        if (QFileInfo(url).exists()) {
            device = new QFile(url);
            device->open(QIODevice::ReadOnly);
        } else {
            if (!nam.hasLocalData())
                nam.setLocalData(new QNetworkAccessManager());
            const QUrl qURL(url, QUrl::StrictMode);
            if (qURL.isValid() && !qURL.isRelative()) {
                QNetworkRequest req = QNetworkRequest(qURL);
                req.setRawHeader("User-Agent", "br");
                QNetworkReply *reply = nam.localData()->get(req);

                reply->waitForReadyRead(-1);
                while (!reply->isFinished())
                    QCoreApplication::processEvents();

                if (reply->error() != QNetworkReply::NoError) {
                    qDebug() << reply->errorString() << url;
                    reply->deleteLater();
                } else {
                    device = reply;
                }
            }
        }

        QByteArray data;
        if (device) {
            data = device->readAll();
            delete device;
            device = NULL;
        }

        if (!data.isEmpty()) {
            Mat encoded(1, data.size(), CV_8UC1, (void*)data.data());
            encoded = encoded.clone();
            if (mode == Permissive) {
                dst += encoded;
            } else {
                Mat decoded = imdecode(encoded, IMREAD_UNCHANGED);
                if (!decoded.empty())
                    dst += (mode == Encoded) ? encoded : decoded;
            }

            dst.file.set("ImageID", QVariant(QCryptographicHash::hash(data, QCryptographicHash::Md5).toHex()));
            dst.file.set("AlgorithmID", data.isEmpty() ? 0 : (mode == Decoded ? 5 : 3));
        } else {
            dst.file.fte = true;
            qWarning("Error opening %s", qPrintable(url));
        }
    }
};
QThreadStorage<QNetworkAccessManager*> DownloadTransform::nam;

BR_REGISTER(Transform, DownloadTransform)

} // namespace br

#include "io/download.moc"
