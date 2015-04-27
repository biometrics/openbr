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
        Template t;
        br_universal_template ut;
        if (gallery.read((char*)&ut, sizeof(br_universal_template)) == sizeof(br_universal_template)) {
            QByteArray data(ut.urlSize + ut.fvSize, Qt::Uninitialized);
            char *dst = data.data();
            qint64 bytesNeeded = ut.urlSize + ut.fvSize;
            while (bytesNeeded > 0) {
                qint64 bytesRead = gallery.read(dst, bytesNeeded);
                if (bytesRead <= 0) {
                    qDebug() << gallery.errorString();
                    qFatal("Unexepected EOF while reading universal template data, needed: %d more of: %d bytes.", int(bytesNeeded), int(ut.urlSize + ut.fvSize));
                }
                bytesNeeded -= bytesRead;
                dst += bytesRead;
            }

            t.file.set("ImageID", QVariant(QByteArray((const char*)ut.imageID, 16).toHex()));
            t.file.set("AlgorithmID", ut.algorithmID);
            t.file.set("URL", QString(data.data()));
            char *dataStart = data.data() + ut.urlSize;
            uint32_t dataSize = ut.fvSize;
            if ((ut.algorithmID <= -1) && (ut.algorithmID >= -3)) {
                t.file.set("FrontalFace", QRectF(ut.x, ut.y, ut.width, ut.height));
                uint32_t *rightEyeX = reinterpret_cast<uint32_t*>(dataStart);
                dataStart += sizeof(uint32_t);
                uint32_t *rightEyeY = reinterpret_cast<uint32_t*>(dataStart);
                dataStart += sizeof(uint32_t);
                uint32_t *leftEyeX = reinterpret_cast<uint32_t*>(dataStart);
                dataStart += sizeof(uint32_t);
                uint32_t *leftEyeY = reinterpret_cast<uint32_t*>(dataStart);
                dataStart += sizeof(uint32_t);
                dataSize -= sizeof(uint32_t)*4;
                t.file.set("First_Eye", QPointF(*rightEyeX, *rightEyeY));
                t.file.set("Second_Eye", QPointF(*leftEyeX, *leftEyeY));
            }
            else if (ut.algorithmID == 7) {
                // binary data consisting of a single channel matrix, of a supported type.
                // 4 element header:
                // uint16 datatype (single channel opencv datatype code)
                // uint32 matrix rows
                // uint32 matrix cols
                // uint16 matrix depth (max 512)
                // Followed by serialized data, in row-major order (in r/c), with depth values
                // for each layer listed in order (i.e. rgb, rgb etc.)
                // #### NOTE! matlab's default order is col-major, so some work should
                // be done on the matlab side to make sure that the initial serialization is correct.
                uint16_t dataType = *reinterpret_cast<uint32_t*>(dataStart);
                dataStart += sizeof(uint16_t);

                uint32_t matrixRows = *reinterpret_cast<uint32_t*>(dataStart);
                dataStart += sizeof(uint32_t);

                uint32_t matrixCols = *reinterpret_cast<uint32_t*>(dataStart);
                dataStart += sizeof(uint32_t);

                uint16_t matrixDepth= *reinterpret_cast<uint16_t*>(dataStart);
                dataStart += sizeof(uint16_t);

                // Set metadata
                t.file.set("Label", ut.label);
                t.file.set("X", ut.x);
                t.file.set("Y", ut.y);
                t.file.set("Width", ut.width);
                t.file.set("Height", ut.height);

                t.append(cv::Mat(matrixRows, matrixCols, CV_MAKETYPE(dataType, matrixDepth), dataStart).clone() /* We don't want a shallow copy! */);
                return t;
            }
            else {
                t.file.set("X", ut.x);
                t.file.set("Y", ut.y);
                t.file.set("Width", ut.width);
                t.file.set("Height", ut.height);
            }
            t.file.set("Label", ut.label);
            t.append(cv::Mat(1, dataSize, CV_8UC1, dataStart).clone() /* We don't want a shallow copy! */);
        } else {
            if (!gallery.atEnd())
                qWarning("Failed to read universal template header!");
            gallery.close();
        }
        return t;
    }

    void writeTemplate(const Template &t)
    {
        const QByteArray imageID = QByteArray::fromHex(t.file.get<QByteArray>("ImageID", QByteArray(32, '0')));
        if (imageID.size() != 16)
            qFatal("Expected 16-byte ImageID, got: %d bytes.", imageID.size());

        const int32_t algorithmID = (t.isEmpty() || t.file.fte) ? 0 : t.file.get<int32_t>("AlgorithmID");

        // QUrl::fromUserInput provides some nice functionality in terms of completing URLs
        // e.g. C:/test.jpg -> file://C:/test.jpg and google.com/image.jpg -> http://google.com/image.jpg
        const QByteArray url = QUrl::fromUserInput(t.file.get<QString>("URL", t.file.name)).toEncoded();

        int32_t x = 0, y = 0;
        uint32_t width = 0, height = 0;
        QByteArray header;
        if ((algorithmID <= -1) && (algorithmID >= -3)) {
            const QRectF frontalFace = t.file.get<QRectF>("FrontalFace");
            x      = frontalFace.x();
            y      = frontalFace.y();
            width  = frontalFace.width();
            height = frontalFace.height();

            const QPointF firstEye   = t.file.get<QPointF>("First_Eye");
            const QPointF secondEye  = t.file.get<QPointF>("Second_Eye");
            const uint32_t rightEyeX = firstEye.x();
            const uint32_t rightEyeY = firstEye.y();
            const uint32_t leftEyeX  = secondEye.x();
            const uint32_t leftEyeY  = secondEye.y();

            header.append((const char*)&rightEyeX, sizeof(uint32_t));
            header.append((const char*)&rightEyeY, sizeof(uint32_t));
            header.append((const char*)&leftEyeX , sizeof(uint32_t));
            header.append((const char*)&leftEyeY , sizeof(uint32_t));
        } else {
            x = t.file.get<int32_t>("X", 0);
            y = t.file.get<int32_t>("Y", 0);
            width = t.file.get<uint32_t>("Width", 0);
            height = t.file.get<uint32_t>("Height", 0);
        }
        const uint32_t label = t.file.get<uint32_t>("Label", 0);

        gallery.write(imageID);
        gallery.write((const char*) &algorithmID, sizeof(int32_t));
        gallery.write((const char*) &x          , sizeof(int32_t));
        gallery.write((const char*) &y          , sizeof(int32_t));
        gallery.write((const char*) &width      , sizeof(uint32_t));
        gallery.write((const char*) &height     , sizeof(uint32_t));
        gallery.write((const char*) &label      , sizeof(uint32_t));

        const uint32_t urlSize = url.size() + 1;
        gallery.write((const char*) &urlSize, sizeof(uint32_t));

        const uint32_t signatureSize = (algorithmID == 0) ? 0 : t.m().rows * t.m().cols * t.m().elemSize();
        const uint32_t fvSize = header.size() + signatureSize;
        gallery.write((const char*) &fvSize, sizeof(uint32_t));

        gallery.write((const char*) url.data(), urlSize);
        if (algorithmID != 0) {
            gallery.write(header);
            gallery.write((const char*) t.m().data, signatureSize);
        }
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
class jsonGallery : public BinaryGallery
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
            qWarning("Couldn't parse: %s\n", line.data());
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

BR_REGISTER(Gallery, jsonGallery)

} // namespace br

#include "gallery/binary.moc"
