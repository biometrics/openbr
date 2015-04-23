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

#include <fstream>
#include <QQueue>
#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \brief DOCUMENT ME
 * \author Unknown \cite unknown
 */
class seqGallery : public Gallery
{
public:
    Q_OBJECT

    bool open()
    {
        seqFile.open(QtUtils::getAbsolutePath(file.name).toStdString().c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if (!isOpen()) {
            qDebug("Failed to open file %s for reading", qPrintable(file.name));
            return false;
        }

        int headSize = 1024;
        // start at end of file to get full size
        int fileSize = seqFile.tellg();
        if (fileSize < headSize) {
            qDebug("No header in seq file");
            return false;
        }

        // first 4 bytes store 0xEDFE, next 24 store 'Norpix seq  '
        char firstFour[4];
        seqFile.seekg(0, std::ios::beg);
        seqFile.read(firstFour, 4);
        char nextTwentyFour[24];
        readText(24, nextTwentyFour);
        if (firstFour[0] != (char)0xED || firstFour[1] != (char)0xFE || strncmp(nextTwentyFour, "Norpix seq", 10) != 0) {
            qDebug("Invalid header in seq file");
            return false;
        }

        // next 8 bytes for version (skipped below) and header size (1024), then 512 for descr
        seqFile.seekg(4, std::ios::cur);
        int hSize = readInt();
        if (hSize != headSize) {
            qDebug("Invalid header size");
            return false;
        }
        char desc[512];
        readText(512, desc);
        file.set("Description", QString(desc));

        width = readInt();
        height = readInt();
        // get # channels from bit depth
        numChan = readInt()/8;
        int imageBitDepthReal = readInt();
        if (imageBitDepthReal != 8) {
            qDebug("Invalid bit depth");
            return false;
        }
        // the size of just the image part of a raw img
        imgSizeBytes = readInt();

        int imgFormatInt = readInt();
        if (imgFormatInt == 100 || imgFormatInt == 200 || imgFormatInt == 101) {
            imgFormat = "raw";
        } else if (imgFormatInt == 102 || imgFormatInt == 201 || imgFormatInt == 103 ||
                   imgFormatInt == 1 || imgFormatInt == 2) {
            imgFormat = "compressed";
        } else {
            qFatal("unsupported image format");
        }

        numFrames = readInt();
        // skip empty int
        seqFile.seekg(4, std::ios::cur);
        // the size of a full raw file, with extra crap after img data
        trueImgSizeBytes = readInt();

        // gather all the frame positions in an array
        seekPos.reserve(numFrames);
        // start at end of header
        seekPos.append(headSize);
        // extra 8 bytes at end of img
        int extra = 8;
        for (int i=1; i<numFrames; i++) {
            int s;
            // compressed images have different sizes
            // the first byte at the beginning of the file
            // says how big the current img is
            if (imgFormat == "compressed") {
                int lastPos = seekPos[i-1];
                seqFile.seekg(lastPos, std::ios::beg);
                int currSize = readInt();
                s = lastPos + currSize + extra;

                // but there might be 16 extra bytes instead of 8...
                if (i == 1) {
                    seqFile.seekg(s, std::ios::beg);
                    char zero;
                    seqFile.read(&zero, 1);
                    if (zero == 0) {
                        s += 8;
                        extra += 8;
                    }
                }
            }
            // raw images are all the same size
            else {
                s = headSize + (i*trueImgSizeBytes);
            }

            seekPos.enqueue(s);
        }

#ifdef CVMATIO
        if (basis.file.contains("vbb")) {
            QString vbb = basis.file.get<QString>("vbb");
            annotations = TemplateList::fromGallery(File(vbb));
        }
#else
        qWarning("cvmatio not installed, bounding boxes will not be available. Add -DBR_WITH_CVMATIO cmake flag to install.");
#endif

        return true;
    }

    bool isOpen()
    {
        return seqFile.is_open();
    }

    void close()
    {
        seqFile.close();
    }

    TemplateList readBlock(bool *done)
    {
        if (!isOpen()) {
            if (!open())
                qFatal("Failed to open file %s for reading", qPrintable(file.name));
            else
                idx = 0;
        }

        // if we've reached the last frame, we're done
        if (seekPos.size() == 0) {
            *done = true;
            return TemplateList();
        }

        seqFile.seekg(seekPos.dequeue(), std::ios::beg);

        cv::Mat temp;
        // let imdecode do all the work to decode the compressed img
        if (imgFormat == "compressed") {
            int imgSize = readInt() - 4;
            std::vector<char> imgBuf(imgSize);
            seqFile.read(&imgBuf[0], imgSize);
            // flags < 0 means load image as-is (keep color info if available)
            cv::imdecode(imgBuf, -1, &temp);
        }
        // raw images can be loaded straight into a Mat
        else {
            char *imgBuf = new char[imgSizeBytes];
            seqFile.read(imgBuf, imgSizeBytes);
            int type = (numChan == 1 ? CV_8UC1 : CV_8UC3);
            temp = cv::Mat(height, width, type, imgBuf);
        }
        Template output;
        output.file = file;
        if (!annotations.empty()) {
            output.file.setRects(annotations.first().file.rects());
            annotations.removeFirst();
        }
        output.m() = temp;
        output.file.set("position",idx);
        idx++;

        *done = false;
        TemplateList rVal;
        rVal.append(output);

        return rVal;
    }

    void write(const Template &t)
    {
        (void) t;
        qFatal("Not implemented.");
    }

private:
    qint64 idx;
    int readInt()
    {
        int num;
        seqFile.read((char*)&num, 4);
        return num;
    }

    // apparently the text in seq files is 16 bit characters (UTF-16?)
    // since we don't really need the last byte, snad since it gets interpreted as
    // a terminating char, let's just grab the first byte for storage
    void readText(int bytes, char *buffer)
    {
        seqFile.read(buffer, bytes);
        for (int i=0; i<bytes; i+=2) {
            buffer[i/2] = buffer[i];
        }
        buffer[bytes/2] = '\0';
    }

protected:
    std::ifstream seqFile;
    QQueue<int> seekPos;
    int width, height, numChan, imgSizeBytes, trueImgSizeBytes, numFrames;
    QString imgFormat;
    TemplateList annotations;
};

BR_REGISTER(Gallery, seqGallery)

} // namespace br

#include "gallery/seq.moc"
