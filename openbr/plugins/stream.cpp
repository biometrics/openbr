#include <fstream>
#include <QReadWriteLock>
#include <QWaitCondition>
#include <QThreadPool>
#include <QSemaphore>
#include <QMap>
#include <QQueue>
#include <QtConcurrent>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include "openbr_internal.h"
#include "openbr/core/common.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/qtutils.h"

using namespace cv;
using namespace std;

namespace br
{

class Idiocy : public QObject
{
    Q_OBJECT
public:
    enum StreamModes { StreamVideo,
                     DistributeFrames,
                     StreamGallery,
                     Auto};

    Q_ENUMS(StreamModes)
};

class FrameData
{
public:
    int sequenceNumber;
    TemplateList data;
};

// A buffer shared between adjacent processing stages in a stream
class SharedBuffer
{
public:
    SharedBuffer() {}
    virtual ~SharedBuffer() {}

    virtual void addItem(FrameData * input)=0;
    virtual void reset()=0;

    virtual FrameData * tryGetItem()=0;
    virtual int size()=0;
};

// for n - 1 boundaries, multiple threads call addItem, the frames are
// sequenced based on FrameData::sequence_number, and calls to getItem
// receive them in that order
class SequencingBuffer : public SharedBuffer
{
public:
    SequencingBuffer()
    {
        next_target = 0;
    }

    void addItem(FrameData * input)
    {
        QMutexLocker bufferLock(&bufferGuard);

        buffer.insert(input->sequenceNumber, input);
    }

    FrameData * tryGetItem()
    {
        QMutexLocker bufferLock(&bufferGuard);

        if (buffer.empty() || buffer.begin().key() != this->next_target) {
            return NULL;
        }

        QMap<int, FrameData *>::Iterator result = buffer.begin();

        if (next_target != result.value()->sequenceNumber) {
            qFatal("mismatched targets!");
        }

        next_target = next_target + 1;

        FrameData * output = result.value();
        buffer.erase(result);
        return output;
    }

    virtual int size()
    {
        QMutexLocker lock(&bufferGuard);
        return buffer.size();
    }
    virtual void reset()
    {
        if (size() != 0)
            qDebug("Sequencing buffer has non-zero size during reset!");

        QMutexLocker lock(&bufferGuard);
        next_target = 0;
    }


private:
    QMutex bufferGuard;
    int next_target;

    QMap<int, FrameData *> buffer;
};

// For 1 - 1 boundaries, a double buffering scheme
// Producer/consumer read/write from separate buffers, and switch if their
// buffer runs out/overflows. Synchronization is handled by a read/write lock
// threads are "reading" if they are adding to/removing from their individual
// buffer, and writing if they access or swap with the other buffer.
class DoubleBuffer : public SharedBuffer
{
public:
    DoubleBuffer()
    {
        inputBuffer = &buffer1;
        outputBuffer = &buffer2;
    }

    int size()
    {
        QReadLocker readLock(&bufferGuard);
        return inputBuffer->size() + outputBuffer->size();
    }

    // called from the producer thread
    void addItem(FrameData * input)
    {
        QReadLocker readLock(&bufferGuard);
        inputBuffer->append(input);
    }

    FrameData * tryGetItem()
    {
        QReadLocker readLock(&bufferGuard);

        // There is something for us to get
        if (!outputBuffer->empty()) {
            FrameData * output = outputBuffer->first();
            outputBuffer->removeFirst();
            return output;
        }

        // Outputbuffer is empty, try to swap with the input buffer, we need a
        // write lock to do that.
        readLock.unlock();
        QWriteLocker writeLock(&bufferGuard);

        // Nothing on the input buffer either?
        if (inputBuffer->empty()) {
            return NULL;
        }

        // input buffer is non-empty, so swap the buffers
        std::swap(inputBuffer, outputBuffer);

        // Return a frame
        FrameData * output = outputBuffer->first();
        outputBuffer->removeFirst();
        return output;
    }

    virtual void reset()
    {
        if (this->size() != 0)
            qDebug("Shared buffer has non-zero size during reset!");
    }


private:
    // The read-write lock. The thread adding to this buffer can add
    // to the current input buffer if it has a read lock. The thread
    // removing from this buffer can remove things from the current
    // output buffer if it has a read lock, or swap the buffers if it
    // has a write lock.
    QReadWriteLock bufferGuard;

    // The buffer that is currently being added to
    QList<FrameData *> * inputBuffer;
    // The buffer that is currently being removed from
    QList<FrameData *> * outputBuffer;

    // The buffers pointed at by inputBuffer/outputBuffer
    QList<FrameData *> buffer1;
    QList<FrameData *> buffer2;
};

// Given a template as input, return N templates as output, one at a time on subsequent
// calls to getNext
class TemplateProcessor
{
public:
    virtual ~TemplateProcessor() {}
    virtual bool open(Template & input)=0;
    virtual bool isOpen()=0;
    virtual void close()=0;
    virtual bool getNextTemplate(Template & output)=0;
protected:
    Template basis;
    string getAbsolutePath(QString filename)
    {
        // Yes, we should specify absolute path:
        // http://stackoverflow.com/questions/9396459/loading-a-video-in-opencv-in-python
        QString fileName = (Globals->path.isEmpty() ? "" : Globals->path + "/") + filename;
        return QFileInfo(fileName).absoluteFilePath().toStdString();
    }
};

static QMutex openLock;

// Read a video frame by frame using cv::VideoCapture
class VideoReader : public TemplateProcessor
{
public:
    VideoReader() {}

    bool open(Template &input)
    {
        basis = input;

        // We can open either files (well actually this includes addresses of ip cameras
        // through ffmpeg), or webcams. Webcam VideoCaptures are created through a separate
        // overload of open that takes an integer, not a string.
        // So, does this look like an integer?
        bool is_int = false;
        int anInt = input.file.name.toInt(&is_int);
        if (is_int)
        {
            bool rc = video.open(anInt);

            if (!rc)
            {
                qDebug("open failed!");
            }
            if (!video.isOpened())
            {
                qDebug("Video not open!");
            }
        } else {
            // On windows, this appears to not be thread-safe
            QMutexLocker lock(&openLock);
            video.open(getAbsolutePath(input.file.name));
        }

        return video.isOpened();
    }

    bool isOpen() { return video.isOpened(); }

    void close() { video.release(); }

    bool getNextTemplate(Template & output)
    {
        if (!isOpen()) {
            qDebug("video source is not open");
            return false;
        }
        output.file = basis.file;
        output.m() = cv::Mat();

        cv::Mat temp;
        bool res = video.read(temp);

        if (!res) {
            // The video capture broke, return false.
            output.m() = cv::Mat();
            close();
            return false;
        }

        // This clone is critical, if we don't do it then the matrix will
        // be an alias of an internal buffer of the video source, leading
        // to various problems later.
        output.m() = temp.clone();
        return true;
    }
protected:
    cv::VideoCapture video;
};


class StreamGallery : public TemplateProcessor
{
public:
    StreamGallery()
    {

    }

    bool open(Template &input)
    {
        // Create a gallery
        gallery = QSharedPointer<Gallery>(Gallery::make(input.file));
        // Failed ot open the gallery?
        if (gallery.isNull()) {
            qDebug()<<"Failed to create gallery!";
            galleryOk = false;
            return false;
        }

        // Set up state variables for future reads
        galleryOk = true;
        gallery->set_readBlockSize(100);
        nextIdx = 0;
        lastBlock = false;
        return galleryOk;
    }

    bool isOpen() { return galleryOk; }

    void close()
    {
        galleryOk = false;
        currentData.clear();
        nextIdx = 0;
        lastBlock = true;
    }

    bool getNextTemplate(Template & output)
    {
        // If we still have data available, we return one of those
        if (nextIdx >= currentData.size())
        {
            // Otherwise, read another block
            if (!lastBlock) {
                currentData = gallery->readBlock(&lastBlock);
                nextIdx = 0;
            }
            else
            {
                galleryOk = false;
                return false;
            }
        }
        // Return the indicated template, and advance the index
        output = currentData[nextIdx++];
        return true;
    }

protected:

    QSharedPointer<Gallery> gallery;
    bool galleryOk;
    bool lastBlock;

    TemplateList currentData;
    int nextIdx;

};

class DirectReturn : public TemplateProcessor
{
public:
    DirectReturn()
    {
        data_ok = false;
    }

    // We don't do anything, just prepare to return input when getNext is called.
    bool open(Template &input)
    {
        basis = input;
        data_ok =true;
        return data_ok;
    }

    bool isOpen() { return data_ok; }

    void close()
    {
        data_ok = false;
        basis.clear();
    }

    bool getNextTemplate(Template & output)
    {
        if (!data_ok)
            return false;
        output = basis;
        data_ok = false;
        return true;
    }

protected:
    // Have we sent our template yet?
    bool data_ok;
};

class SeqReader : public TemplateProcessor
{
public:
    SeqReader() {}

    bool open(Template &input)
    {
        basis = input;

        seqFile.open(getAbsolutePath(input.file.name).c_str(), ios::in | ios::binary | ios::ate);
        if (!isOpen()) return false;

        int headSize = 1024;
        // start at end of file to get full size
        int fileSize = seqFile.tellg();
        if (fileSize < headSize) {
            qDebug("No header in seq file");
            return false;
        }

        // first 4 bytes store 0xEDFE, next 24 store 'Norpix seq  '
        char firstFour[4];
        seqFile.seekg(0, ios::beg);
        seqFile.read(firstFour, 4);
        char nextTwentyFour[24];
        readText(24, nextTwentyFour);
        if (firstFour[0] != (char)0xED || firstFour[1] != (char)0xFE || strncmp(nextTwentyFour, "Norpix seq", 10) != 0) {
            qDebug("Invalid header in seq file");
            return false;
        }

        // next 8 bytes for version (skipped below) and header size (1024), then 512 for descr
        seqFile.seekg(4, ios::cur);
        int hSize = readInt();
        if (hSize != headSize) {
            qDebug("Invalid header size");
            return false;
        }
        char desc[512];
        readText(512, desc);
        basis.file.set("Description", QString(desc));

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
        seqFile.seekg(4, ios::cur);
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
                seqFile.seekg(lastPos, ios::beg);
                int currSize = readInt();
                s = lastPos + currSize + extra;

                // but there might be 16 extra bytes instead of 8...
                if (i == 1) {
                    seqFile.seekg(s, ios::beg);
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

    bool getNextTemplate(Template &output)
    {
        if (!isOpen()) {
            qDebug("Seq not open");
            return false;
        }
        // if we've reached the last frame, we're done
        if (seekPos.size() == 0) return false;

        seqFile.seekg(seekPos.dequeue(), ios::beg);

        Mat temp;
        // let imdecode do all the work to decode the compressed img
        if (imgFormat == "compressed") {
            int imgSize = readInt() - 4;
            vector<char> imgBuf(imgSize);
            seqFile.read(&imgBuf[0], imgSize);
            // flags < 0 means load image as-is (keep color info if available)
            imdecode(imgBuf, -1, &temp);
        }
        // raw images can be loaded straight into a Mat
        else {
            char *imgBuf = new char[imgSizeBytes];
            seqFile.read(imgBuf, imgSizeBytes);
            int type = (numChan == 1 ? CV_8UC1 : CV_8UC3);
            temp = Mat(height, width, type, imgBuf);
        }

        output.file = basis.file;
        if (!annotations.empty()) {
            output.file.setRects(annotations.first().file.rects());
            annotations.removeFirst();
        }
        output.m() = temp;

        return true;
    }
private:
    int readInt()
    {
        int num;
        seqFile.read((char*)&num, 4);
        return num;
    }

    // apparently the text in seq files is 16 bit characters (UTF-16?)
    // since we don't really need the last byte, snad since it gets interpreted as
    // a terminating char, let's just grab the first byte for storage
    void readText(int bytes, char * buffer)
    {
        seqFile.read(buffer, bytes);
        for (int i=0; i<bytes; i+=2) {
            buffer[i/2] = buffer[i];
        }
        buffer[bytes/2] = '\0';
    }

protected:
    ifstream seqFile;
    QQueue<int> seekPos;
    int width, height, numChan, imgSizeBytes, trueImgSizeBytes, numFrames;
    QString imgFormat;
    TemplateList annotations;
};

// Interface for sequentially getting data from some data source.
// Given a TemplateList, return single template frames sequentially by applying a TemplateProcessor
// to each individual template.
class DataSource
{
public:
    DataSource(int maxFrames=500)
    {
        // The sequence number of the last frame
        final_frame = -1;
        for (int i=0; i < maxFrames;i++)
        {
            allFrames.addItem(new FrameData());
        }
        frameSource = NULL;
    }

    virtual ~DataSource()
    {
        while (true)
        {
            FrameData * frame = allFrames.tryGetItem();
            if (frame == NULL)
                break;
            delete frame;
        }
    }

    void close()
    {
        if (this->frameSource)
        {
            frameSource->close();
            delete frameSource;
            frameSource = NULL;
        }
    }

    int size()
    {
        return this->templates.size();
    }

    bool open(const TemplateList & input, br::Idiocy::StreamModes _mode)
    {
        // Set up variables specific to us
        current_template_idx = 0;
        templates = input;
        mode = _mode;

        is_broken = false;
        allReturned = false;

        // The last frame isn't initialized yet
        final_frame = -1;
        // Start our sequence numbers from the input index
        next_sequence_number = 0;

        // Actually open the data source
        bool open_res = openNextTemplate();

        // We couldn't open the data source
        if (!open_res) {
            is_broken = true;
            return false;
        }

        // Try to get a frame from the global pool
        FrameData * firstFrame = allFrames.tryGetItem();

        // If this fails, things have gone pretty badly.
        if (firstFrame == NULL) {
            is_broken = true;
            return false;
        }

        // Read a frame from the video source
        bool res = getNextFrame(*firstFrame);

        // the data source broke already, we couldn't even get one frame
        // from it even though it claimed to have opened successfully.
        if (!res) {
            is_broken = true;
            return false;
        }

        // We read one frame ahead of the last one returned, this allows
        // us to know which frame is the final frame when we return it.
        lookAhead.append(firstFrame);
        return true;
    }


    // non-blocking version of getFrame
    // Returns a NULL FrameData if too many frames are out, or the
    // data source is broken. Sets last_frame to true iff the FrameData
    // returned is the last valid frame, and the data source is now broken.
    FrameData * tryGetFrame(bool & last_frame)
    {
        last_frame = false;

        if (is_broken) {
            return NULL;
        }

        // Try to get a FrameData from the pool, if we can't it means too many
        // frames are already out, and we will return NULL to indicate failure
        FrameData * aFrame = allFrames.tryGetItem();
        if (aFrame == NULL)
            return NULL;

        // Try to actually read a frame, if this returns false the data source is broken
        bool res = getNextFrame(*aFrame);

        // The datasource broke, update final_frame
        if (!res)
        {
            QMutexLocker lock(&last_frame_update);
            final_frame = lookAhead.back()->sequenceNumber;
            allFrames.addItem(aFrame);
        }
        else {
            lookAhead.push_back(aFrame);
        }

        // we will return the first frame on the lookAhead buffer
        FrameData * rVal = lookAhead.first();
        lookAhead.pop_front();
        if (rVal->data.empty())
            qDebug("returning empty frame from look ahead!");

        // If this is the last frame, say so
        if (rVal->sequenceNumber == final_frame) {
            last_frame = true;
            is_broken = true;
        }

        return rVal;
    }

    // Return a frame to the pool, returns true if the frame returned was the last
    // frame issued, false otherwise
    bool returnFrame(FrameData * inputFrame)
    {
        int frameNumber = inputFrame->sequenceNumber;

        inputFrame->data.clear();
        inputFrame->sequenceNumber = -1;
        allFrames.addItem(inputFrame);

        bool rval = false;

        QMutexLocker lock(&last_frame_update);

        if (frameNumber == final_frame) {
            // We just received the last frame, better pulse
            allReturned = true;
            rval = true;
        }

        return rval;
    }

    void wake()
    {
        lastReturned.wakeAll();
    }

    bool waitLast()
    {
        QMutexLocker lock(&last_frame_update);

        while (!allReturned)
        {
            // This would be a safer wait if we used a timeout, but
            // theoretically that should never matter.
            lastReturned.wait(&last_frame_update);
        }
        return true;
    }

protected:

    bool openNextTemplate()
    {
        if (this->current_template_idx >= this->templates.size())
            return false;

        bool open_res = false;
        while (!open_res)
        {
            if (frameSource)
                frameSource->close();

            Template curr = this->templates[current_template_idx];
            if (mode == br::Idiocy::Auto)
            {
                delete frameSource;
                if (curr.empty()) {
                    if (curr.file.name.right(3) == "seq")
                        frameSource = new SeqReader();
                    else
                        frameSource = new VideoReader();
                }
                else
                    frameSource = new DirectReturn();
            }
            else if (mode == br::Idiocy::DistributeFrames)
            {
                if (!frameSource)
                    frameSource = new DirectReturn();
            }
            else if (mode == br::Idiocy::StreamGallery)
            {
                if (!frameSource)
                    frameSource = new StreamGallery();
            }
            else if (mode == br::Idiocy::StreamVideo)
            {
                if (!frameSource) {
                    if (curr.file.name.right(3) == "seq")
                        frameSource = new SeqReader();
                    else
                        frameSource = new VideoReader();
                }
            }
            open_res = frameSource->open(curr);
            if (!open_res)
            {
                current_template_idx++;
                if (current_template_idx >= this->templates.size())
                    return false;
            }
        }
        return true;
    }

    bool getNextFrame(FrameData & output)
    {
        bool got_frame = false;

        Template aTemplate;

        while (!got_frame)
        {
            got_frame = frameSource->getNextTemplate(aTemplate);

            // OK we got a frame
            if (got_frame) {
                // set the sequence number and tempalte of this frame
                output.sequenceNumber = next_sequence_number;
                output.data.append(aTemplate);
                // set the frame number in the template's metadata
                output.data.last().file.set("FrameNumber", output.sequenceNumber);
                next_sequence_number++;
                return true;
            }

            // advance to the next tempalte in our list
            this->current_template_idx++;
            bool open_res = this->openNextTemplate();

            // couldn't get the next template? nothing to do, otherwise we try to read
            // a frame at the top of this loop.
            if (!open_res) {
                return false;
            }
        }

        return false;
    }

    // Index of the template in the templatelist we are currently reading from
    int current_template_idx;

    // What do we do to each template
    br::Idiocy::StreamModes mode;

    // list of templates we are workign from
    TemplateList templates;

    // processor for the current template
    TemplateProcessor * frameSource;

    int next_sequence_number;
    int final_frame;
    bool is_broken;
    bool allReturned;

    DoubleBuffer allFrames;
    QList<FrameData *> lookAhead;

    QWaitCondition lastReturned;
    QMutex last_frame_update;
};

class ProcessingStage;

class BasicLoop : public QRunnable, public QFutureInterface<void>
{
public:
    BasicLoop()
    {
        this->reportStarted();
    }

    void run();

    QList<ProcessingStage *> * stages;
    int start_idx;
    FrameData * startItem;
};

class ProcessingStage
{
public:
    friend class DirectStreamTransform;
public:
    ProcessingStage(int nThreads = 1)
    {
        thread_count = nThreads;
    }
    virtual ~ProcessingStage() {}

    virtual FrameData* run(FrameData * input, bool & should_continue, bool & final)=0;

    virtual bool tryAcquireNextStage(FrameData *& input, bool & final)=0;

    int stage_id;

    virtual void reset()=0;

    virtual void status()=0;

protected:
    int thread_count;

    SharedBuffer * inputBuffer;
    ProcessingStage * nextStage;
    QList<ProcessingStage *> * stages;
    QThreadPool * threads;
    Transform * transform;

};

class MultiThreadStage : public ProcessingStage
{
public:
    MultiThreadStage(int _input) : ProcessingStage(_input) {}

    // Not much to worry about here, we will project the input
    // and try to continue to the next stage.
    FrameData * run(FrameData * input, bool & should_continue, bool & final)
    {
        if (input == NULL) {
            qFatal("null input to multi-thread stage");
        }

        input->data >> *transform;

        should_continue = nextStage->tryAcquireNextStage(input, final);

        return input;
    }

    // Called from a different thread than run. Nothing to worry about
    // we offer no restrictions on when loops may enter this stage.
    virtual bool tryAcquireNextStage(FrameData *& input, bool & final)
    {
        (void) input;
        final = false;
        return true;
    }

    void reset()
    {
        // nothing to do.
    }
    void status(){
        qDebug("multi thread stage %d, nothing to worry about", this->stage_id);
    }
};

class SingleThreadStage : public ProcessingStage
{
public:
    SingleThreadStage(bool input_variance) : ProcessingStage(1)
    {
        currentStatus = STOPPING;
        next_target = 0;
        // If the previous stage is single-threaded, queued inputs
        // are stored in a double buffer
        if (input_variance) {
            this->inputBuffer = new DoubleBuffer();
        }
        // If it's multi-threaded we need to put the inputs back in order
        // before we can use them, so we use a sequencing buffer.
        else {
            this->inputBuffer = new SequencingBuffer();
        }
    }

    ~SingleThreadStage()
    {
        delete inputBuffer;
    }

    void reset()
    {
        QWriteLocker writeLock(&statusLock);
        currentStatus = STOPPING;
        next_target = 0;
        inputBuffer->reset();
    }


    int next_target;
    enum Status
    {
        STARTING,
        STOPPING
    };
    QReadWriteLock statusLock;
    Status currentStatus;

    FrameData * run(FrameData * input, bool & should_continue, bool & final)
    {
        if (input == NULL)
            qFatal("NULL input to stage %d", this->stage_id);

        if (input->sequenceNumber != next_target)
        {
            qFatal("out of order frames for stage %d, got %d expected %d", this->stage_id, input->sequenceNumber, this->next_target);
        }
        next_target = input->sequenceNumber + 1;

        // Project the input we got
        transform->projectUpdate(input->data);

        should_continue = nextStage->tryAcquireNextStage(input,final);

        if (final)
            return input;

        // Is there anything on our input buffer? If so we should start a thread with that.
        QWriteLocker lock(&statusLock);
        FrameData * newItem = inputBuffer->tryGetItem();
        if (!newItem)
        {
            this->currentStatus = STOPPING;
        }
        lock.unlock();

        if (newItem)
            startThread(newItem);

        return input;
    }

    void startThread(br::FrameData * newItem)
    {
        BasicLoop * next = new BasicLoop();
        next->stages = stages;
        next->start_idx = this->stage_id;
        next->startItem = newItem;

        // We start threads with priority equal to their stage id
        // This is intended to ensure progression, we do queued late stage
        // jobs before queued early stage jobs, and so tend to finish frames
        // rather than go stage by stage. In Qt 5.1, priorities are priorities
        // so we use the stage_id directly.
        this->threads->start(next, stage_id);
    }


    // Calledfrom a different thread than run.
    bool tryAcquireNextStage(FrameData *& input, bool & final)
    {
        final = false;
        inputBuffer->addItem(input);

        QReadLocker lock(&statusLock);
        // Thread is already running, we should just return
        if (currentStatus == STARTING)
        {
            return false;
        }
        // Have to change to a write lock to modify currentStatus
        lock.unlock();

        QWriteLocker writeLock(&statusLock);
        // But someone else might have started a thread in the meantime
        if (currentStatus == STARTING)
        {
            return false;
        }
        // Ok we might start a thread, as long as we can get something back
        // from the input buffer
        input = inputBuffer->tryGetItem();

        if (!input)
            return false;

        currentStatus = STARTING;

        return true;
    }

    void status(){
        qDebug("single thread stage %d, status starting? %d, next %d buffer size %d", this->stage_id, this->currentStatus == SingleThreadStage::STARTING, this->next_target, this->inputBuffer->size());
    }

};

// Semi-functional, doesn't do anything productive outside of stream::train
class CollectSets : public TimeVaryingTransform
{
    Q_OBJECT
public:
    CollectSets() : TimeVaryingTransform(false, false) {}

    QList<TemplateList> sets;

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        (void) dst;
        sets.append(src);
    }

    void train(const TemplateList & data)
    {
        (void) data;
    }

};

// This stage reads new frames from the data source.
class ReadStage : public SingleThreadStage
{
public:
    ReadStage(int activeFrames = 100) : SingleThreadStage(true), dataSource(activeFrames){ }

    DataSource dataSource;

    void reset()
    {
        dataSource.close();
        SingleThreadStage::reset();
    }

    FrameData * run(FrameData * input, bool & should_continue, bool & final)
    {
        if (input == NULL)
            qFatal("NULL frame in input stage");

        // Can we enter the next stage?
        should_continue = nextStage->tryAcquireNextStage(input, final);

        // Try to get a frame from the datasource, we keep working on
        // the frame we have, but we will queue another job for the next
        // frame if a frame is currently available.
        QWriteLocker lock(&statusLock);
        bool last_frame = false;
        FrameData * newFrame = dataSource.tryGetFrame(last_frame);

        // Were we able to get a frame?
        if (newFrame) startThread(newFrame);
        // If not this stage will enter a stopped state.
        else {
            currentStatus = STOPPING;
        }

        lock.unlock();

        return input;
    }

    // The last stage, trying to access the first stage
    bool tryAcquireNextStage(FrameData *& input, bool & final)
    {
        // Return the frame, was it the last one?
        final = dataSource.returnFrame(input);
        input = NULL;

        // OK we won't continue.
        if (final) {
            return false;
        }

        QReadLocker lock(&statusLock);
        // If the first stage is already active we will just end.
        if (currentStatus == STARTING)
        {
            return false;
        }

        // Otherwise we will try to continue, but to do so we have to
        // escalate the lock, and sadly there is no way to do so without
        // releasing the read-mode lock, and getting a new write-mode lock.
        lock.unlock();

        QWriteLocker writeLock(&statusLock);
        // currentStatus might have changed in the gap between releasing the read
        // lock and getting the write lock.
        if (currentStatus == STARTING)
        {
            return false;
        }

        bool last_frame = false;
        // Try to get a frame from the data source, if we get one we will
        // continue to the first stage.
        input = dataSource.tryGetFrame(last_frame);

        if (!input) {
            return false;
        }

        currentStatus = STARTING;

        return true;
    }

    void status(){
        qDebug("Read stage %d, status starting? %d, next frame %d buffer size %d", this->stage_id, this->currentStatus == SingleThreadStage::STARTING, this->next_target, this->dataSource.size());
    }
};

void BasicLoop::run()
{
    int current_idx = start_idx;
    FrameData * target_item = startItem;
    bool should_continue = true;
    bool the_end = false;
    forever
    {
        target_item = stages->at(current_idx)->run(target_item, should_continue, the_end);
        if (!should_continue) {
            break;
        }
        current_idx++;
        current_idx = current_idx % stages->size();
    }
    if (the_end) {
        dynamic_cast<ReadStage *> (stages->at(0))->dataSource.wake();
    }

    this->reportFinished();

}

class DirectStreamTransform : public CompositeTransform
{
    Q_OBJECT
public:

    Q_PROPERTY(int activeFrames READ get_activeFrames WRITE set_activeFrames RESET reset_activeFrames)
    Q_PROPERTY(br::Idiocy::StreamModes readMode READ get_readMode WRITE set_readMode RESET reset_readMode)
    BR_PROPERTY(int, activeFrames, 100)
    BR_PROPERTY(br::Idiocy::StreamModes, readMode, br::Idiocy::Auto)

    friend class StreamTransfrom;

    void subProject(QList<TemplateList> & data, int end_idx)
    {
        if (end_idx == 0)
            return;

        CollectSets collector;

        // Set transforms to the start, up to end_idx
        QList<Transform *> backup = this->transforms;
        transforms = backup.mid(0,end_idx);
        // We use collector to retain the project structure at the end of the
        // truncated stream.
        transforms.append(&collector);

        // Reinitialize, we now act as a shorter stream.
        init();

        QList<TemplateList> output;
        for (int i=0; i < data.size(); i++) {
            projectUpdate(data[i], data[i]);
            output.append(collector.sets);
            collector.sets.clear();
        }
        data = output;
        transforms = backup;
    }

    void train(const QList<TemplateList> & data)
    {
        if (!trainable) {
            qWarning("Attempted to train untrainable transform, nothing will happen.");
            return;
        }
        QList<TemplateList> separated;
        foreach (const TemplateList & list, data) {
            foreach(const Template & t, list) {
                separated.append(TemplateList());
                separated.last().append(t);
            }
        }

        for (int i=0; i < transforms.size(); i++) {
            // OK we have a trainable transform, we need to get input data for it.
            if (transforms[i]->trainable) {
                QList<TemplateList> copy = separated;
                // Project from the start to the trainable stage.
                subProject(copy,i);

                transforms[i]->train(copy);
            }
        }
        // Re-initialize because subProject probably messed us up.
        init();
    }

    bool timeVarying() const { return true; }

    void project(const Template &src, Template &dst) const
    {
        TemplateList in;
        in.append(src);
        TemplateList out;
        CompositeTransform::project(in,out);
        dst = out.first();
        if (out.size() > 1)
            qDebug("Returning first output template only");
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        TemplateList in;
        in.append(src);
        TemplateList out;
        projectUpdate(in,out);
        dst = out.first();
        if (out.size() > 1)
            qDebug("Returning first output template only");
    }


    virtual void finalize(TemplateList & output)
    {
        (void) output;
        // Nothing in particular to do here, stream calls finalize
        // on all child transforms as part of projectUpdate
    }

    // start processing, consider all templates in src a continuous
    // 'video'
    void projectUpdate(const TemplateList & src, TemplateList & dst)
    {
        dst = src;
        if (src.empty())
            return;

        bool res = readStage->dataSource.open(src,readMode);
        if (!res) {
            qDebug("stream failed to open %s", qPrintable(dst[0].file.name));
            return;
        }

        // Start the first thread in the stream.
        QWriteLocker lock(&readStage->statusLock);
        readStage->currentStatus = SingleThreadStage::STARTING;

        // We have to get a frame before starting the thread
        bool last_frame = false;
        FrameData * firstFrame = readStage->dataSource.tryGetFrame(last_frame);
        if (firstFrame == NULL)
            qFatal("Failed to read first frame of video");

        readStage->startThread(firstFrame);
        lock.unlock();

        // Wait for the stream to process the last frame available from
        // the data source.
        readStage->dataSource.waitLast();

        // Now that there are no more incoming frames, call finalize
        // on each transform in turn to collect any last templates
        // they wish to issue.
        TemplateList final_output;

        // Push finalize through the stages
        for (int i=0; i < this->transforms.size(); i++)
        {
            TemplateList output_set;
            transforms[i]->finalize(output_set);
            if (output_set.empty())
                continue;

            for (int j=i+1; j < transforms.size();j++)
            {
                transforms[j]->projectUpdate(output_set);
            }
            final_output.append(output_set);
        }

        // Clear dst, since we set it to src so that the datasource could open it
        dst.clear();

        // dst is set to all output received by the final stage, along
        // with anything output via the calls to finalize.
        foreach(const TemplateList & list, collector->sets) {
            dst.append(list);
        }
        collector->sets.clear();

        dst.append(final_output);

        foreach(ProcessingStage * stage, processingStages) {
            stage->reset();
        }
    }


    // Create and link stages
    void init()
    {
        if (transforms.isEmpty()) return;

        for (int i=0; i < processingStages.size();i++)
            delete processingStages[i];
        processingStages.clear();

        // call CompositeTransform::init so that trainable is set
        // correctly.
        CompositeTransform::init();

        // We share a thread pool across streams attached to the same
        // parent tranform, retrieve or create a thread pool based
        // on our parent transform.
        QMutexLocker poolLock(&poolsAccess);
        QHash<QObject *, QThreadPool *>::Iterator it;
        if (!pools.contains(this->parent())) {
            it = pools.insert(this->parent(), new QThreadPool(this->parent()));
            it.value()->setMaxThreadCount(Globals->parallelism);
        }
        else it = pools.find(this->parent());
        threads = it.value();
        poolLock.unlock();

        // Are our children time varying or not? This decides whether
        // we run them in single threaded or multi threaded stages
        stage_variance.clear();
        stage_variance.reserve(transforms.size());
        foreach (const br::Transform *transform, transforms) {
            stage_variance.append(transform->timeVarying());
        }

        // Additionally, we have a separate stage responsible for reading
        // frames from the data source
        readStage = new ReadStage(activeFrames);

        processingStages.push_back(readStage);
        readStage->stage_id = 0;
        readStage->stages = &this->processingStages;
        readStage->threads = this->threads;

        // Initialize and link a processing stage for each of our child
        // transforms.
        int next_stage_id = 1;
        bool prev_stage_variance = true;
        for (int i =0; i < transforms.size(); i++)
        {
            if (stage_variance[i])
                // Whether or not the previous stage is multi-threaded controls
                // the type of input buffer we need in a single threaded stage.
                processingStages.append(new SingleThreadStage(prev_stage_variance));
            else
                processingStages.append(new MultiThreadStage(Globals->parallelism));

            processingStages.last()->stage_id = next_stage_id++;

            // link nextStage pointers, the stage we just appeneded is i+1 since
            // the read stage was added before this loop
            processingStages[i]->nextStage = processingStages[i+1];

            processingStages.last()->stages = &this->processingStages;
            processingStages.last()->threads = this->threads;

            processingStages.last()->transform = transforms[i];
            prev_stage_variance = stage_variance[i];
        }

        // We also have the last stage, which just puts the output of the
        // previous stages on a template list.
        collectionStage = new SingleThreadStage(prev_stage_variance);
        collectionStage->transform = this->collector.data();


        processingStages.append(collectionStage);
        collectionStage->stage_id = next_stage_id;
        collectionStage->stages = &this->processingStages;
        collectionStage->threads = this->threads;

        // the last transform stage points to collection stage
        processingStages[processingStages.size() - 2]->nextStage = collectionStage;

        // And the collection stage points to the read stage, because this is
        // a ring buffer.
        collectionStage->nextStage = readStage;
    }

    DirectStreamTransform()
    {
        this->collector = QSharedPointer<CollectSets>(new CollectSets());
    }

    ~DirectStreamTransform()
    {
        // Delete all the stages
        for (int i = 0; i < processingStages.size(); i++) {
// TODO: Are we releasing memory which is already freed?
            delete processingStages[i];
        }
        processingStages.clear();
    }

protected:
    QList<bool> stage_variance;

    ReadStage * readStage;
    SingleThreadStage * collectionStage;
    QSharedPointer<CollectSets> collector;

    QList<ProcessingStage *> processingStages;

    // This is a map from parent transforms (of Streams) to thread pools. Rather
    // than starting threads on the global thread pool, Stream uses separate thread pools
    // keyed on their parent transform. This is necessary because stream's project starts
    // threads, then enters an indefinite wait for them to finish. Since we are starting
    // threads using thread pools, threads themselves are a limited resource. Therefore,
    // the type of hold and wait done by stream project can lead to deadlock unless
    // resources are ordered in such a way that a circular wait will not occur. The points
    // of this hash is to introduce a resource ordering (on threads) that mirrors the structure
    // of the algorithm. So, as long as the structure of the algorithm is a DAG, the wait done
    // by stream project will not be circular, since every thread in stream project is waiting
    // for threads at a lower level to do the work.
    // This issue doesn't come up in distribute, since a thread waiting on a QFutureSynchronizer
    // will steal work from those jobs, so in that sense distribute isn't doing a hold and wait.
    // Waiting for a QFutureSynchronzier isn't really possible here since stream runs an indeteriminate
    // number of jobs.
    static QHash<QObject *, QThreadPool *> pools;
    static QMutex poolsAccess;
    QThreadPool * threads;

    void _project(const Template &src, Template &dst) const
    {
        (void) src; (void) dst;
        qFatal("nope");
    }
    void _project(const TemplateList & src, TemplateList & dst) const
    {
        (void) src; (void) dst;
        qFatal("nope");
    }
};

QHash<QObject *, QThreadPool *> DirectStreamTransform::pools;
QMutex DirectStreamTransform::poolsAccess;

BR_REGISTER(Transform, DirectStreamTransform)

class StreamTransform : public WrapperTransform
{
    Q_OBJECT

public:
    StreamTransform() : WrapperTransform(false)
    {
    }

    Q_PROPERTY(int activeFrames READ get_activeFrames WRITE set_activeFrames RESET reset_activeFrames)
    Q_PROPERTY(br::Idiocy::StreamModes readMode READ get_readMode WRITE set_readMode RESET reset_readMode)

    BR_PROPERTY(int, activeFrames, 100)
    BR_PROPERTY(br::Idiocy::StreamModes, readMode, br::Idiocy::Auto)

    bool timeVarying() const { return true; }

    void project(const Template &src, Template &dst) const
    {
        basis.project(src,dst);
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        basis.projectUpdate(src,dst);
    }
    void projectUpdate(const TemplateList & src, TemplateList & dst)
    {
        basis.projectUpdate(src,dst);
    }

    void train(const QList<TemplateList> & data)
    {
        basis.train(data);
    }

    virtual void finalize(TemplateList & output)
    {
        (void) output;
        // Nothing in particular to do here, stream calls finalize
        // on all child transforms as part of projectUpdate
    }

    // reinterpret transform, set up the actual stream. We can only reinterpret pipes
    void init()
    {
        if (!transform)
            return;

        trainable = transform->trainable;

        basis.setParent(this->parent());
        basis.transforms.clear();
        basis.activeFrames = this->activeFrames;
        basis.readMode = this->readMode;

        // We need at least a CompositeTransform * to acess transform's children.
        CompositeTransform * downcast = dynamic_cast<CompositeTransform *> (transform);

        // If this isn't even a composite transform, or it's not a pipe, just set up
        // basis with 1 stage.
        if (!downcast || QString(transform->metaObject()->className()) != "br::PipeTransform")
        {
            basis.transforms.append(transform);
            basis.init();
            return;
        }
        if (downcast->transforms.empty())
        {
            qWarning("Trying to set up empty stream");
            basis.init();
            return;
        }

        // OK now we will regroup downcast's children
        QList<QList<Transform *> > sets;
        sets.append(QList<Transform *> ());
        sets.last().append(downcast->transforms[0]);
        if (downcast->transforms[0]->timeVarying())
            sets.append(QList<Transform *> ());

        for (int i=1;i < downcast->transforms.size(); i++) {
            // If this is time varying it becomse its own stage
            if (downcast->transforms[i]->timeVarying()) {
                // If a set was already active, we add another one
                if (!sets.last().empty()) {
                    sets.append(QList<Transform *>());
                }
                // add the item
                sets.last().append(downcast->transforms[i]);
                // Add another set to indicate separation.
                sets.append(QList<Transform *>());
            }
            // otherwise, we can combine non time-varying stages
            else {
                sets.last().append(downcast->transforms[i]);
            }

        }
        if (sets.last().empty())
            sets.removeLast();

        QList<Transform *> transform_set;
        transform_set.reserve(sets.size());
        for (int i=0; i < sets.size(); i++) {
            // If this is a single transform set, we add that to the list
            if (sets[i].size() == 1 ) {
                transform_set.append(sets[i].at(0));
            }
            //otherwise we build a pipe
            else {
                CompositeTransform * pipe = dynamic_cast<CompositeTransform *>(Transform::make("Pipe([])", this));
                pipe->transforms = sets[i];
                pipe->init();
                transform_set.append(pipe);
            }
        }

        basis.transforms = transform_set;
        basis.init();
    }

    Transform * smartCopy(bool & newTransform)
    {
        // We just want the DirectStream to begin with, so just return a copy of that.
        DirectStreamTransform * res = (DirectStreamTransform *) basis.smartCopy(newTransform);
        res->activeFrames = this->activeFrames;
        return res;
    }


private:
    DirectStreamTransform basis;
};

BR_REGISTER(Transform, StreamTransform)

} // namespace br

#include "stream.moc"

