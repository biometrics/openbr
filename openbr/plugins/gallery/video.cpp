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

#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \brief Read a video frame by frame using cv::VideoCapture
 * \author Unknown \cite unknown
 */
class videoGallery : public Gallery
{
    Q_OBJECT
public:
    qint64 idx;
    ~videoGallery()
    {
        video.release();
    }

    static QMutex openLock;

    virtual void deferredInit()
    {
        bool status = video.open(QtUtils::getAbsolutePath(file.name).toStdString());

        if (!status)
            qFatal("Failed to open file %s with path %s", qPrintable(file.name), qPrintable(QtUtils::getAbsolutePath(file.name)));
    }

    TemplateList readBlock(bool *done)
    {
        if (!video.isOpened()) {
            // opening videos appears to not be thread safe on windows
            QMutexLocker lock(&openLock);

            deferredInit();
            idx = 0;
        }

        Template output;
        output.file = file;
        output.m() = cv::Mat();

        cv::Mat temp;
        bool res = video.read(temp);

        if (!res) {
            // The video capture broke, return an empty list.
            output.m() = cv::Mat();
            video.release();
            *done = true;
            return TemplateList();
        }

        // This clone is critical, if we don't do it then the output matrix will
        // be an alias of an internal buffer of the video source, leading to various
        // problems later.
        output.m() = temp.clone();

        output.file.set("progress", idx);
        idx++;

        TemplateList rVal;
        rVal.append(output);
        *done = false;
        return rVal;
    }

    void write(const Template &t)
    {
        (void)t; qFatal("Not implemented");
    }

protected:
    cv::VideoCapture video;
};

BR_REGISTER(Gallery,videoGallery)

QMutex videoGallery::openLock;

/*!
 * \brief Read videos of format .avi
 * \author Unknown \cite unknown
 */
class aviGallery : public videoGallery
{
    Q_OBJECT
};

BR_REGISTER(Gallery, aviGallery)

/*!
 * \brief Read videos of format .wmv
 * \author Unknown \cite unknown
 */
class wmvGallery : public videoGallery
{
    Q_OBJECT
};

BR_REGISTER(Gallery, wmvGallery)

/*!
 * \brief Read a video from the webcam
 * \author Unknown \cite unknown
 */
class webcamGallery : public videoGallery
{
public:
    Q_OBJECT

    void deferredInit()
    {
        bool intOK = false;
        int anInt = file.baseName().toInt(&intOK);

        if (!intOK)
            qFatal("Expected integer basename, got %s", qPrintable(file.baseName()));

        bool rc = video.open(anInt);

        if (!rc)
            qFatal("Failed to open webcam with index: %s", qPrintable(file.baseName()));
    }

};

BR_REGISTER(Gallery,webcamGallery)

} // namespace br

#include "gallery/video.moc"
