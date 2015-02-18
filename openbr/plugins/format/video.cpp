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
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup formats
 * \brief Read all frames of a video using OpenCV
 * \author Charles Otto \cite caotto
 */
class videoFormat : public Format
{
    Q_OBJECT

public:
    Template read() const
    {
        if (!file.exists() )
            return Template();

        VideoCapture videoSource(file.name.toStdString());
        videoSource.open(file.name.toStdString() );


        Template frames;
        if (!videoSource.isOpened()) {
            qWarning("video file open failed");
            return frames;
        }

        bool open = true;
        while (open) {
            cv::Mat frame;
            open = videoSource.read(frame);
            if (!open) break;

            frames.append(cv::Mat());
            frames.back() = frame.clone();
        }

        return frames;
    }

    void write(const Template &t) const
    {
        int fourcc = OpenCVUtils::getFourcc();
        VideoWriter videoSink(file.name.toStdString(), fourcc, 30, t.begin()->size());

        // Did we successfully open the output file?
        if (!videoSink.isOpened() ) qFatal("Failed to open output file");

        for (Template::const_iterator it = t.begin(); it!= t.end(); ++it) {
            videoSink << *it;
        }
    }
};

BR_REGISTER(Format, videoFormat)

/*!
 * \ingroup formats
 * \brief Retrieves an image from a webcam.
 * \author Josh Klontz \cite jklontz
 */
class webcamFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        static QScopedPointer<VideoCapture> videoCapture;

        if (videoCapture.isNull())
            videoCapture.reset(new VideoCapture(0));

        Mat m;
        videoCapture->read(m);
        return Template(m);
    }

    void write(const Template &t) const
    {
        (void) t;
        qFatal("Not supported.");
    }
};

BR_REGISTER(Format, webcamFormat)

/*!
 * \ingroup formats
 * \brief Reads image files.
 * \author Josh Klontz \cite jklontz
 */
class DefaultFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        Template t;

        if (file.name.startsWith("http://") || file.name.startsWith("https://") || file.name.startsWith("www.")) {
            if (Factory<Format>::names().contains("url")) {
                File urlFile = file;
                urlFile.name.append(".url");
                QScopedPointer<Format> url(Factory<Format>::make(urlFile));
                t = url->read();
            }
        } else {
            Mat m = imread(file.resolved().toStdString());
            if (m.data) {
                t.append(m);
            } else {
                videoFormat videoReader;
                videoReader.file = file;
                t = videoReader.read();
            }
        }

        return t;
    }

    void write(const Template &t) const
    {
        if (t.size() > 1) {
            videoFormat videoWriter;
            videoWriter.file = file;
            videoWriter.write(t);
        } else if (t.size() == 1) {
            QtUtils::touchDir(QDir(file.path()));
            imwrite(file.name.toStdString(), t);
        }
    }
};

BR_REGISTER(Format, DefaultFormat)

} // namespace br

#include "format/video.moc"
