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

#ifndef BR_EMBEDDED
#include <QtXml>
#endif // BR_EMBEDDED
#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup formats
 * \brief Decodes images from Base64 xml
 * \author Scott Klum \cite sklum
 * \author Josh Klontz \cite jklontz
 */
class xmlFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        Template t;

#ifndef BR_EMBEDDED
        QString fileName = file.get<QString>("path") + file.name;

        QDomDocument doc(fileName);
        QFile f(fileName);

        if (!f.open(QIODevice::ReadOnly)) qFatal("Unable to open %s for reading.", qPrintable(file.flat()));
        if (!doc.setContent(&f)) qWarning("Unable to parse %s.", qPrintable(file.flat()));
        f.close();

        QDomElement docElem = doc.documentElement();
        QDomNode subject = docElem.firstChild();
        while (!subject.isNull()) {
            QDomNode fileNode = subject.firstChild();

            while (!fileNode.isNull()) {
                QDomElement e = fileNode.toElement();

                if (e.tagName() == "FORMAL_IMG") {
                    QByteArray byteArray = QByteArray::fromBase64(qPrintable(e.text()));
                    Mat m = imdecode(Mat(3, byteArray.size(), CV_8UC3, byteArray.data()), CV_LOAD_IMAGE_COLOR);
                    if (!m.data) qWarning("xmlFormat::read failed to decode image data.");
                    t.append(m);
                } else if ((e.tagName() == "RELEASE_IMG") ||
                           (e.tagName() == "PREBOOK_IMG") ||
                           (e.tagName() == "LPROFILE") ||
                           (e.tagName() == "RPROFILE")) {
                    // Ignore these other image fields for now
                } else {
                    t.file.set(e.tagName(), e.text());
                }

                fileNode = fileNode.nextSibling();
            }
            subject = subject.nextSibling();
        }

        // Calculate age
        if (t.file.contains("DOB")) {
            const QDate dob = QDate::fromString(t.file.get<QString>("DOB").left(10), "yyyy-MM-dd");
            const QDate current = QDate::currentDate();
            int age = current.year() - dob.year();
            if (current.month() < dob.month()) age--;
            t.file.set("Age", age);
        }
#endif // BR_EMBEDDED

        return t;
    }

    void write(const Template &t) const
    {
        QStringList lines;
        lines.append("<?xml version=\"1.0\" standalone=\"yes\"?>");
        lines.append("<openbr-xml-format>");
        lines.append("\t<xml-data>");
        foreach (const QString &key, t.file.localKeys()) {
            if ((key == "Index") || (key == "Label")) continue;
            lines.append("\t\t<"+key+">"+QtUtils::toString(t.file.value(key))+"</"+key+">");
        }
        std::vector<uchar> data;
        imencode(".jpg",t.m(),data);
        QByteArray byteArray = QByteArray::fromRawData((const char*)data.data(), data.size());
        lines.append("\t\t<FORMAL_IMG>"+byteArray.toBase64()+"</FORMAL_IMG>");
        lines.append("\t</xml-data>");
        lines.append("</openbr-xml-format>");
        QtUtils::writeFile(file, lines);
    }
};

BR_REGISTER(Format, xmlFormat)

} // namespace br

#include "format/xml.moc"
