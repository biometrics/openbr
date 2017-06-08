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

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup formats
 * \brief MATLAB <tt>.mat</tt> format.
 *
 * matFormat is known not to work with compressed matrices
 *
 * \author Josh Klontz \cite jklontz
 * \br_link http://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf
 */
class matFormat : public Format
{
    Q_OBJECT

    struct Element
    {
        // It is always best to cast integers to a Qt integer type, such as qint16 or quint32, when reading and writing.
        // This ensures that you always know exactly what size integers you are reading and writing, no matter what the
        // underlying platform and architecture the application happens to be running on.
        // http://qt-project.org/doc/qt-4.8/datastreamformat.html
        quint32 type, bytes;
        QByteArray data;
        Element() : type(0), bytes(0) {}
        Element(QDataStream &stream)
            : type(0), bytes(0)
        {
            // Read first 4 bytes into type (32 bit integer),
            // specifying the type of data used
            if (stream.readRawData((char*)&type, 4) != 4)
                qFatal("Unexpected end of file.");

            if (type >= 1 << 16) {
                // Small data format
                bytes = type;
                type = type & 0x0000FFFF;
                bytes = bytes >> 16;
            } else {
                // Regular format
                // Read 4 bytes into bytes (32 bit integer),
                // specifying the size of the element
                if (stream.readRawData((char*)&bytes, 4) != 4)
                    qFatal("Unexpected end of file.");
            }

            // Set the size of data to bytes
            data.resize(bytes);

            // Read bytes amount of data from the file into data
            if (int(bytes) != stream.readRawData(data.data(), bytes))
                qFatal("Unexpected end of file.");

            // Alignment
            int skipBytes = (bytes < 4) ? (4 - bytes) : (8 - bytes%8)%8;
            if (skipBytes != 0) stream.skipRawData(skipBytes);
        }
    };

    Template read() const
    {
        QByteArray byteArray;
        QtUtils::readFile(file, byteArray);
        QDataStream f(byteArray);

        { // Check header
            QByteArray header(128, 0);
            f.readRawData(header.data(), 128);
            if (!header.startsWith("MATLAB 5.0 MAT-file"))
                qFatal("Invalid MAT header.");
        }

        Template t(file);

        while (!f.atEnd()) {
            Element element(f);

            // miCOMPRESSED
            if (element.type == 15) {
                // Prepend the number of bytes to element.data
                element.data.prepend((char*)&element.bytes, 4); // Qt zlib wrapper requires this to preallocate the buffer
                QDataStream uncompressed(qUncompress(element.data));
                element = Element(uncompressed);
            }

            // miMATRIX
            if (element.type == 14) {
                QDataStream matrix(element.data);
                qint32 rows = 0, columns = 0;
                int matrixType = 0;
                QByteArray matrixData;
                while (!matrix.atEnd()) {
                    Element subelement(matrix);
                    if (subelement.type == 5) { // Dimensions array
                        if (subelement.bytes == 8) {
                            rows = ((qint32*)subelement.data.constData())[0];
                            columns = ((qint32*)subelement.data.constData())[1];
                        } else {
                            qWarning("matFormat::read can only handle 2D arrays.");
                        }
                    } else if (subelement.type == 7) { //miSINGLE
                        matrixType = CV_32FC1;
                        matrixData = subelement.data;
                    } else if (subelement.type == 9) { //miDOUBLE
                        matrixType = CV_64FC1;
                        matrixData = subelement.data;
                    }
                }

                if ((rows > 0) && (columns > 0) && (matrixType != 0) && !matrixData.isEmpty()) {
                    Mat transposed;
                    transpose(Mat(columns, rows, matrixType, matrixData.data()), transposed);
                    t.append(transposed);
                }
            }
        }

        return t;
    }

    void write(const Template &t) const
    {
        QByteArray data;
        QDataStream stream(&data, QFile::WriteOnly);

        { // Header
            QByteArray header = "MATLAB 5.0 MAT-file; Made with OpenBR | www.openbiometrics.org\n";
            QByteArray buffer(116-header.size(), 0);
            stream.writeRawData(header.constData(), header.size());
            stream.writeRawData(buffer.constData(), buffer.size());
            quint64 subsystem = 0;
            quint16 version = 0x0100;
            const char *endianness = "IM";
            stream.writeRawData((const char*)&subsystem, 8);
            stream.writeRawData((const char*)&version, 2);
            stream.writeRawData(endianness, 2);
        }

        for (int i=0; i<t.size(); i++) {
            const Mat &m = t[i];
            if (m.channels() != 1) qFatal("Only supports single channel matrices.");

            QByteArray subdata;
            QDataStream substream(&subdata, QFile::WriteOnly);

            {  // Array Flags
                quint32 type = 6;
                quint32 bytes = 8;
                quint64 arrayClass = 0;
                switch (m.type()) {
                  case CV_64FC1: arrayClass = 6; break;
                  case CV_32FC1: arrayClass = 7; break;
                  case CV_8UC1: arrayClass = 8; break;
                  case CV_8SC1: arrayClass = 9; break;
                  case CV_16UC1: arrayClass = 10; break;
                  case CV_16SC1: arrayClass = 11; break;
                  case CV_32SC1: arrayClass = 12; break;
                  default: qFatal("Unsupported matrix class.");
                }
                substream.writeRawData((const char*)&type, 4);
                substream.writeRawData((const char*)&bytes, 4);
                substream.writeRawData((const char*)&arrayClass, 8);
            }

            { // Dimensions Array
                quint32 type = 5;
                quint32 bytes = 8;
                substream.writeRawData((const char*)&type, 4);
                substream.writeRawData((const char*)&bytes, 4);
                substream.writeRawData((const char*)&m.rows, 4);
                substream.writeRawData((const char*)&m.cols, 4);
            }

            { // Array Name
                QByteArray name(qPrintable(QString("OpenBR_%1").arg(QString::number(i))));
                quint32 type = 1;
                quint32 bytes = name.size();
                QByteArray buffer((8 - bytes%8)%8, 0);
                substream.writeRawData((const char*)&type, 4);
                substream.writeRawData((const char*)&bytes, 4);
                substream.writeRawData(name.constData(), name.size());
                substream.writeRawData(buffer.constData(), buffer.size());
            }

            { // Real part
                quint32 type = 0;
                switch (m.type()) {
                  case CV_8SC1:  type = 1; break;
                  case CV_8UC1:  type = 2; break;
                  case CV_16SC1: type = 3; break;
                  case CV_16UC1: type = 4; break;
                  case CV_32SC1: type = 5; break;
                  case CV_32FC1: type = 7; break;
                  case CV_64FC1: type = 9; break;
                  default: qFatal("Unsupported matrix type.");
                }
                quint32 bytes = m.elemSize() * m.rows * m.cols;
                QByteArray buffer((8 - bytes%8)%8, 0);
                Mat transposed;
                transpose(m, transposed);
                substream.writeRawData((const char*)&type, 4);
                substream.writeRawData((const char*)&bytes, 4);
                substream.writeRawData((const char*)transposed.data, bytes);
                substream.writeRawData(buffer.constData(), buffer.size());
            }

            { // Matrix
                quint32 type = 14;
                quint32 bytes = subdata.size();
                stream.writeRawData((const char*)&type, 4);
                stream.writeRawData((const char*)&bytes, 4);
                stream.writeRawData(subdata.constData(), subdata.size());
            }
        }

        QtUtils::writeFile(file, data);
    }
};

BR_REGISTER(Format, matFormat)

} // namespace br

#include "format/mat.moc"
