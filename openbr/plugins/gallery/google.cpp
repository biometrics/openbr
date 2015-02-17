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

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup inputs
 * \brief Input from a google image search.
 * \author Josh Klontz \cite jklontz
 */
class googleGallery : public Gallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        TemplateList templates;

        static const QString search = "http://images.google.com/images?q=%1&start=%2";
        QString query = file.name.left(file.name.size()-7); // remove ".google"

#ifndef BR_EMBEDDED
        QNetworkAccessManager networkAccessManager;
        for (int i=0; i<100; i+=20) { // Retrieve 100 images
            QNetworkRequest request(search.arg(query, QString::number(i)));
            QNetworkReply *reply = networkAccessManager.get(request);

            while (!reply->isFinished())
                QThread::yieldCurrentThread();

            QString data(reply->readAll());
            delete reply;

            QStringList words = data.split("imgurl=");
            words.takeFirst(); // Remove header
            foreach (const QString &word, words) {
                QString url = word.left(word.indexOf("&amp"));
                url = url.replace("%2520","%20");
                int junk = url.indexOf('%', url.lastIndexOf('.'));
                if (junk != -1) url = url.left(junk);
                templates.append(File(url,query));
            }
        }
#endif // BR_EMBEDDED

        *done = true;
        return templates;
    }

    void write(const Template &)
    {
        qFatal("Not supported.");
    }
};

BR_REGISTER(Gallery, googleGallery)

} // namespace br

#include "gallery/google.moc"
