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

#include <QJsonObject>
#include <QJsonParseError>
#include <QUrl>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif // _WIN32

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>
#include <openbr/universal_template.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief An abstract gallery for handling binary data
 * \author Josh Klontz \cite jklontz
 */
class BinaryGallery : public Gallery
{
    Q_OBJECT

    void init()
    {
        const QString baseName = file.baseName();

        if (baseName == "stdin") {
#ifdef _WIN32
            if(_setmode(_fileno(stdin), _O_BINARY) == -1)
                qFatal("Failed to set stdin to binary mode!");
#endif // _WIN32

            gallery.open(stdin, QFile::ReadOnly);
        } else if (baseName == "stdout") {
#ifdef _WIN32
            if(_setmode(_fileno(stdout), _O_BINARY) == -1)
                qFatal("Failed to set stdout to binary mode!");
#endif // _WIN32

            gallery.open(stdout, QFile::WriteOnly);
        } else if (baseName == "stderr") {
#ifdef _WIN32
            if(_setmode(_fileno(stderr), _O_BINARY) == -1)
                qFatal("Failed to set stderr to binary mode!");
#endif // _WIN32

            gallery.open(stderr, QFile::WriteOnly);
        } else {
            // Defer opening the file, in the general case we don't know if we
            // need read or write mode yet
            return;
        }
        stream.setDevice(&gallery);
    }

    void readOpen()
    {
        if (!gallery.isOpen()) {
            gallery.setFileName(file);
            if (!gallery.exists())
                qFatal("File %s does not exist", qPrintable(gallery.fileName()));

            QFile::OpenMode mode = QFile::ReadOnly;
            if (!gallery.open(mode))
                qFatal("Can't open gallery: %s for reading", qPrintable(gallery.fileName()));
            stream.setDevice(&gallery);
        }
    }

    void writeOpen()
    {
        if (!gallery.isOpen()) {
            gallery.setFileName(file);

            // Do we remove the pre-existing gallery?
            if (file.get<bool>("remove"))
                gallery.remove();
            QtUtils::touchDir(gallery);
            QFile::OpenMode mode = QFile::WriteOnly;

            // Do we append?
            if (file.get<bool>("append"))
                mode |= QFile::Append;

            if (!gallery.open(mode))
                qFatal("Can't open gallery: %s for writing", qPrintable(gallery.fileName()));
            stream.setDevice(&gallery);
        }
    }

    TemplateList readBlock(bool *done)
    {
        readOpen();
        if (gallery.atEnd())
            gallery.seek(0);

        TemplateList templates;
        while ((templates.size() < readBlockSize) && !gallery.atEnd()) {
            const Template t = readTemplate();
            if (!t.isEmpty() || !t.file.isNull()) {
                templates.append(t);
                templates.last().file.set("progress", position());
            }

            // Special case for pipes where we want to process data as soon as it is available
            if (gallery.isSequential())
                break;
        }

        *done = gallery.atEnd();
        return templates;
    }

    void write(const Template &t)
    {
        writeOpen();
        writeTemplate(t);
        if (gallery.isSequential())
            gallery.flush();
    }

protected:
    QFile gallery;
    QDataStream stream;

    qint64 totalSize()
    {
        readOpen();
        return gallery.size();
    }

    qint64 position()
    {
        return gallery.pos();
    }

    virtual Template readTemplate() = 0;
    virtual void writeTemplate(const Template &t) = 0;
};

/*!
 * \ingroup galleries
 * \brief A binary gallery.
 *
 * Designed to be a literal translation of templates to disk.
 * Compatible with TemplateList::fromBuffer.
 * \author Josh Klontz \cite jklontz
 */
class galGallery : public BinaryGallery
{
    Q_OBJECT

    Template readTemplate()
    {
        Template t;
        stream >> t;
        return t;
    }

    void writeTemplate(const Template &t)
    {
        if (t.isEmpty() && t.file.isNull())
            return;
        else if (t.file.fte) {
             // Only write metadata for failure to enroll, but remove any stored QVariants of type cv::Mat
            File f = t.file;
            QVariantMap metadata = f.localMetadata();
            QMapIterator<QString, QVariant> i(metadata);
            while (i.hasNext()) {
                i.next();
                if (strcmp(i.value().typeName(),"cv::Mat") == 0)
                    f.remove(i.key());
            }
            stream << Template(f);
        }
        else
            stream << t;
    }
};

BR_REGISTER(Gallery, galGallery)

/*!
 * \ingroup galleries
 * \brief A contiguous array of br_universal_template.
 * \author Josh Klontz \cite jklontz
 */
class utGallery : public BinaryGallery
{
    Q_OBJECT

    Template readTemplate()
    {
        const br_const_utemplate ut = Template::readUniversalTemplate(gallery);
        const Template t = Template::fromUniversalTemplate(ut);
        Template::freeUniversalTemplate(ut);
        return t;
    }

    void writeTemplate(const Template &t)
    {
        const br_utemplate ut = Template::toUniversalTemplate(t);
        gallery.write((const char*) ut, sizeof(br_universal_template) + ut->mdSize + ut->fvSize);
        Template::freeUniversalTemplate(ut);
    }
};

BR_REGISTER(Gallery, utGallery)

/*!
 * \ingroup galleries
 * \brief Newline-separated URLs.
 * \author Josh Klontz \cite jklontz
 */
class urlGallery : public BinaryGallery
{
    Q_OBJECT

    Template readTemplate()
    {
        Template t;
        const QString url = QString::fromLocal8Bit(gallery.readLine()).simplified();
        if (!url.isEmpty())
            t.file.set("URL", url);
        return t;
    }

    void writeTemplate(const Template &t)
    {
        const QString url = t.file.get<QString>("URL", t.file.name);
        if (!url.isEmpty()) {
            gallery.write(qPrintable(url));
            gallery.write("\n");
        }
    }
};

BR_REGISTER(Gallery, urlGallery)

/*!
 * \ingroup galleries
 * \brief Newline-separated JSON objects.
 * \author Josh Klontz \cite jklontz
 */
class jsonObjectGallery : public BinaryGallery
{
    Q_OBJECT

    Template readTemplate()
    {
        QJsonParseError error;
        const QByteArray line = gallery.readLine().simplified();
        if (line.isEmpty())
            return Template();
        File file = QJsonDocument::fromJson(line, &error).object().toVariantMap();
        if (error.error != QJsonParseError::NoError) {
            qWarning("Couldn't parse: %s\n", line.constData());
            qFatal("%s\n", qPrintable(error.errorString()));
        }
        return file;
    }

    void writeTemplate(const Template &t)
    {
        const QByteArray json = QJsonDocument(QJsonObject::fromVariantMap(t.file.localMetadata())).toJson().replace('\n', "");
        if (!json.isEmpty()) {
            gallery.write(json);
            gallery.write("\n");
        }
    }
};

BR_REGISTER(Gallery, jsonObjectGallery)

} // namespace br

#include "gallery/binary.moc"
