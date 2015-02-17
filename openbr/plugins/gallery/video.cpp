#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

// Read a video frame by frame using cv::VideoCapture
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
        rVal.append(temp);
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

class aviGallery : public videoGallery
{
    Q_OBJECT
};

BR_REGISTER(Gallery, aviGallery)

class wmvGallery : public videoGallery
{
    Q_OBJECT
};

BR_REGISTER(Gallery, wmvGallery)

// Mostly the same as videoGallery, but we open the VideoCapture with an integer index
// rather than file name/web address
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
