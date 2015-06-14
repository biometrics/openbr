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

#include <QtXml>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/bee.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief A sigset input.
 * \author Josh Klontz \cite jklontz
 */
class xmlGallery : public FileGallery
{
    Q_OBJECT
    Q_PROPERTY(bool ignoreMetadata READ get_ignoreMetadata WRITE set_ignoreMetadata RESET reset_ignoreMetadata STORED false)
    Q_PROPERTY(bool skipMissing READ get_skipMissing WRITE set_skipMissing RESET reset_skipMissing STORED false)
    BR_PROPERTY(bool, ignoreMetadata, false)
    BR_PROPERTY(bool, skipMissing, false)
    FileList files;

    QXmlStreamReader reader;

    QString currentSignatureName;
    bool signatureActive;

    ~xmlGallery()
    {
        f.close();
        if (!files.isEmpty())
            BEE::writeSigset(file, files, ignoreMetadata);
    }

    TemplateList readBlock(bool *done)
    {
        if (readOpen())
            reader.setDevice(&f);

        if (reader.atEnd())
            f.seek(0);

        TemplateList templates;
        qint64 count = 0;

        while (!reader.atEnd())
        {
            // if an identity is active we try to read presentations
            if (signatureActive)
            {
                while (signatureActive)
                {
                    QXmlStreamReader::TokenType signatureToken = reader.readNext();

                    // did the signature end?
                    if (signatureToken == QXmlStreamReader::EndElement && reader.name() == "biometric-signature") {
                        signatureActive = false;
                        break;
                    }

                    // did we reach the end of the document? Theoretically this shoudln't happen without reaching the end of
                    if (signatureToken == QXmlStreamReader::EndDocument)
                        break;

                    // a presentation!
                    if (signatureToken == QXmlStreamReader::StartElement && reader.name() == "presentation") {
                        templates.append(Template(File("",currentSignatureName)));
                        foreach (const QXmlStreamAttribute &attribute, reader.attributes()) {
                            // file-name is stored directly on file, not as a key/value pair
                            if (attribute.name() == "file-name")
                                templates.last().file.name = attribute.value().toString();
                            // other values are directly set as metadata
                            else if (!ignoreMetadata) templates.last().file.set(attribute.name().toString(), attribute.value().toString());
                        }

                        // a presentation can have bounding boxes as child elements
                        QList<QRectF> rects = templates.last().file.rects();
                        while (true)
                        {
                            QXmlStreamReader::TokenType pToken = reader.readNext();
                            if (pToken == QXmlStreamReader::EndElement && reader.name() == "presentation")
                                break;

                            if (pToken == QXmlStreamReader::StartElement)
                            {
                                if (reader.attributes().hasAttribute("x")
                                    && reader.attributes().hasAttribute("y")
                                    && reader.attributes().hasAttribute("width")
                                    && reader.attributes().hasAttribute("height") )
                                {
                                    // get bounding box properties as attributes, just going to assume this all works
                                    qreal x = reader.attributes().value("x").string()->toDouble();
                                    qreal y = reader.attributes().value("y").string()->toDouble();
                                    qreal width =  reader.attributes().value("width").string()->toDouble();
                                    qreal height = reader.attributes().value("height").string()->toDouble();
                                    rects += QRectF(x, y, width, height);
                                }
                            }
                        }
                        templates.last().file.setRects(rects);
                        templates.last().file.set("progress", f.pos());

                        // we read another complete template
                        count++;

                        // optionally remove templates whose files don't exist or are empty
                        if (skipMissing && !QFileInfo(templates.last().file.resolved()).size()) {
                            templates.removeLast();
                            count--;
                        }
                    }
                }
            }
            // otherwise, keep reading elements until the next identity is reacehed
            else
            {
                QXmlStreamReader::TokenType token = reader.readNext();

                // end of file?
                if (token == QXmlStreamReader::EndDocument)
                    break;

                // we are only interested in new elements
                if (token != QXmlStreamReader::StartElement)
                    continue;

                QStringRef elName = reader.name();

                // biometric-signature-set is the root element
                if (elName == "biometric-signature-set")
                    continue;

                // biometric-signature -- an identity
                if (elName == "biometric-signature")
                {
                    // read the name associated with the current signature
                    if (!reader.attributes().hasAttribute("name"))
                    {
                        qDebug() << "Biometric signature missing name";
                        continue;
                    }
                    currentSignatureName = reader.attributes().value("name").toString();
                    signatureActive = true;

                    // If we've already read enough templates for this block, then break here.
                    // We wait untill the start of the next signature to be sure that done should
                    // actually be false (i.e. there are actually items left in this file)
                    if (count >= this->readBlockSize) {
                        *done = false;
                        return templates;
                    }

                }
            }
        }
        *done = true;

        return templates;
    }

    void write(const Template &t)
    {
        files.append(t.file);
    }

    void init()
    {
        FileGallery::init();
    }
};

BR_REGISTER(Gallery, xmlGallery)

} // namespace br

#include "gallery/xml.moc"
