#include <QReadWriteLock>
#include <QWaitCondition>
#include <QThreadPool>
#include <QSemaphore>
#include <QMap>
#include <opencv/highgui.h>
#include <QtConcurrent>
#include "openbr_internal.h"

#include "openbr/core/common.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/qtutils.h"

using namespace cv;

namespace br
{

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


// Interface for sequentially getting data from some data source.
// Initialized off of a template, can represent a video file (stored in the template's filename)
// or a set of images already loaded into memory stored as multiple matrices in an input template.
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
        bool res = getNext(*aFrame);

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
            lastReturned.wakeAll();
            rval = true;
        }

        return rval;
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

    bool open(Template & output, int start_index = 0)
    {
        is_broken = false;
        allReturned = false;

        // The last frame isn't initialized yet
        final_frame = -1;
        // Start our sequence numbers from the input index
        next_sequence_number = start_index;

        // Actually open the data source
        bool open_res = concreteOpen(output);

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
        bool res = getNext(*firstFrame);

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

    /*
     * Pure virtual methods
     */

    // isOpen doesn't appear to particularly work when used on opencv
    // VideoCaptures, so we don't use it for anything important.
    virtual bool isOpen()=0;
    // Called from open, open the data source specified by the input
    // template, don't worry about setting any of the state variables
    // set in open.
    virtual bool concreteOpen(Template & output) = 0;
    // Get the next frame from the data source, store the results in
    // FrameData (including the actual frame and appropriate sequence
    // number).
    virtual bool getNext(FrameData & input) = 0;
    // close the currently open data source.
    virtual void close() = 0;

    int next_sequence_number;
protected:
    DoubleBuffer allFrames;
    int final_frame;
    bool is_broken;
    bool allReturned;
    QList<FrameData *> lookAhead;

    QWaitCondition lastReturned;
    QMutex last_frame_update;
};

// Read a video frame by frame using cv::VideoCapture
class VideoDataSource : public DataSource
{
public:
    VideoDataSource(int maxFrames) : DataSource(maxFrames) {}

    bool concreteOpen(Template &input)
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
            // Yes, we should specify absolute path:
            // http://stackoverflow.com/questions/9396459/loading-a-video-in-opencv-in-python
            QString fileName = (Globals->path.isEmpty() ? "" : Globals->path + "/") + input.file.name;
            video.open(QFileInfo(fileName).absoluteFilePath().toStdString());
        }

        return video.isOpened();
    }

    bool isOpen() { return video.isOpened(); }

    void close() {
        video.release();
    }

private:
    bool getNext(FrameData & output)
    {
        if (!isOpen()) {
            qDebug("video source is not open");
            return false;
        }

        output.data.append(Template(basis.file));
        output.data.last().m() = cv::Mat();

        output.sequenceNumber = next_sequence_number;
        next_sequence_number++;

        cv::Mat temp;
        bool res = video.read(temp);

        if (!res) {
            // The video capture broke, return false.
            output.data.last().m() = cv::Mat();
            close();
            return false;
        }

        // This clone is critical, if we don't do it then the matrix will
        // be an alias of an internal buffer of the video source, leading
        // to various problems later.
        output.data.last().m() = temp.clone();

        output.data.last().file.set("FrameNumber", output.sequenceNumber);
        return true;
    }

    cv::VideoCapture video;
    Template basis;
};

// Given a template as input, return its matrices one by one on subsequent calls
// to getNext
class TemplateDataSource : public DataSource
{
public:
    TemplateDataSource(int maxFrames) : DataSource(maxFrames)
    {
        current_matrix_idx = INT_MAX;
        data_ok = false;
    }

    // To "open" it we just set appropriate indices, we assume that if this
    // is an image, it is already loaded into memory.
    bool concreteOpen(Template &input)
    {
        basis = input;
        current_matrix_idx = 0;

        data_ok = current_matrix_idx < basis.size();
        return data_ok;
    }

    bool isOpen() {
        return data_ok;
    }

    void close()
    {
        current_matrix_idx = INT_MAX;
        basis.clear();
    }

private:
    bool getNext(FrameData & output)
    {
        data_ok = current_matrix_idx < basis.size();
        if (!data_ok)
            return false;

        output.data.append(basis[current_matrix_idx]);
        current_matrix_idx++;

        output.sequenceNumber = next_sequence_number;
        next_sequence_number++;

        output.data.last().file.set("FrameNumber", output.sequenceNumber);
        return true;
    }

    Template basis;
    // Index of the next matrix to output from the template
    int current_matrix_idx;

    // is current_matrix_idx in bounds?
    bool data_ok;
};

// Given a templatelist as input, create appropriate data source for each
// individual template
class DataSourceManager : public DataSource
{
public:
    DataSourceManager(int activeFrames=100) : DataSource(activeFrames)
    {
        actualSource = NULL;
    }

    ~DataSourceManager()
    {
        close();
    }

    int size()
    {
        return this->allFrames.size();
    }

    void close()
    {
        if (actualSource) {
            actualSource->close();
            delete actualSource;
            actualSource = NULL;
        }
    }

    // We are used through a call to open(TemplateList)
    bool open(TemplateList & input)
    {
        // Set up variables specific to us
        current_template_idx = 0;
        templates = input;

        // Call datasourece::open on the first template to set up
        // state variables
        return DataSource::open(templates[current_template_idx]);
    }

    // Create an actual data source of appropriate type for this template
    // (initially called via the call to DataSource::open, called later
    // as we run out of frames on our templates).
    bool concreteOpen(Template & input)
    {
        close();

        bool open_res = false;
        // Input has no matrices? Its probably a video that hasn't been loaded yet
        if (input.empty()) {
            actualSource = new VideoDataSource(0);
            open_res = actualSource->concreteOpen(input);
        }
        // If the input is not empty, we assume it is a set of frames already
        // in memory.
        else {
            actualSource = new TemplateDataSource(0);
            open_res = actualSource->concreteOpen(input);
        }

        // The data source failed to open
        if (!open_res) {
            delete actualSource;
            actualSource = NULL;
            return false;
        }
        return true;
    }

    bool isOpen() { return !actualSource ? false : actualSource->isOpen(); }

protected:
    // Index of the template in the templatelist we are currently reading from
    int current_template_idx;

    TemplateList templates;
    DataSource * actualSource;
    // Get the next frame, if we run out of frames on the current template
    // move on to the next one.
    bool getNext(FrameData & output)
    {
        bool res = actualSource->getNext(output);
        output.sequenceNumber = next_sequence_number;

        // OK we got a frame
        if (res) {
            // Override the sequence number set by actualSource
            output.data.last().file.set("FrameNumber", output.sequenceNumber);
            next_sequence_number++;
            if (output.data.last().last().empty())
                qDebug("broken matrix");
            return true;
        }

        // We didn't get a frame, try to move on to the next template.
        while(!res) {
            output.data.clear();
            current_template_idx++;

            // No more templates? We're done
            if (current_template_idx >= templates.size())
                return false;

            // open the next data source
            bool open_res = concreteOpen(templates[current_template_idx]);
            // We couldn't open it, give up? We could maybe continue here
            // but don't currently.
            if (!open_res)
                return false;

            // get a frame from the newly opened data source, if that fails
            // we continue to open the next one.
            res = actualSource->getNext(output);
        }
        // Finally, set the sequence number for the frame we actually return.
        output.sequenceNumber = next_sequence_number++;
        output.data.last().file.set("FrameNumber", output.sequenceNumber);

        if (output.data.last().last().empty())
            qDebug("broken matrix");

        return res;
    }

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

    virtual FrameData* run(FrameData * input, bool & should_continue)=0;

    virtual bool tryAcquireNextStage(FrameData *& input)=0;

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

void BasicLoop::run()
{
    int current_idx = start_idx;
    FrameData * target_item = startItem;
    bool should_continue = true;
    forever
    {
        target_item = stages->at(current_idx)->run(target_item, should_continue);
        if (!should_continue) {
            break;
        }
        current_idx++;
        current_idx = current_idx % stages->size();
    }
    this->reportFinished();
}

class MultiThreadStage : public ProcessingStage
{
public:
    MultiThreadStage(int _input) : ProcessingStage(_input) {}

    // Not much to worry about here, we will project the input
    // and try to continue to the next stage.
    FrameData * run(FrameData * input, bool & should_continue)
    {
        if (input == NULL) {
            qFatal("null input to multi-thread stage");
        }
        input->data >> *transform;

        should_continue = nextStage->tryAcquireNextStage(input);

        return input;
    }

    // Called from a different thread than run. Nothing to worry about
    // we offer no restrictions on when loops may enter this stage.
    virtual bool tryAcquireNextStage(FrameData *& input)
    {
        (void) input;
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

    FrameData * run(FrameData * input, bool & should_continue)
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

        should_continue = nextStage->tryAcquireNextStage(input);

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
    bool tryAcquireNextStage(FrameData *& input)
    {
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

// This stage reads new frames from the data source.
class FirstStage : public SingleThreadStage
{
public:
    FirstStage(int activeFrames = 100) : SingleThreadStage(true), dataSource(activeFrames){ }

    DataSourceManager dataSource;

    void reset()
    {
        dataSource.close();
        SingleThreadStage::reset();
    }

    FrameData * run(FrameData * input, bool & should_continue)
    {
        if (input == NULL)
            qFatal("NULL frame in input stage");

        // Can we enter the next stage?
        should_continue = nextStage->tryAcquireNextStage(input);

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
    bool tryAcquireNextStage(FrameData *& input)
    {
        // Return the frame, was it the last one?
        bool was_last = dataSource.returnFrame(input);
        input = NULL;

        // OK we won't continue.
        if (was_last) {
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

// Appened to the end of a Stream's transform sequence. Collects the output
// from each frame on a single templatelist
class LastStage : public SingleThreadStage
{
public:
    LastStage(bool _prev_stage_variance) : SingleThreadStage(_prev_stage_variance) {}
    TemplateList getOutput()
    {
        return collectedOutput;
    }

private:
    TemplateList collectedOutput;
public:

    void reset()
    {
        collectedOutput.clear();
        SingleThreadStage::reset();
    }

    FrameData * run(FrameData * input, bool & should_continue)
    {
        if (input == NULL) {
            qFatal("NULL input to stage %d", this->stage_id);
        }

        if (input->sequenceNumber != next_target)
        {
            qFatal("out of order frames for stage %d, got %d expected %d", this->stage_id, input->sequenceNumber, this->next_target);
        }
        next_target = input->sequenceNumber + 1;

        // add the item to our output buffer
        collectedOutput.append(input->data);

        // Can we enter the read stage?
        should_continue = nextStage->tryAcquireNextStage(input);

        // Is there anything on our input buffer? If so we should start a thread
        // in this stage to process that frame.
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

    void status(){
        qDebug("Collection stage %d, status starting? %d, next %d buffer size %d", this->stage_id, this->currentStatus == SingleThreadStage::STARTING, this->next_target, this->inputBuffer->size());
    }

};



class DirectStreamTransform : public CompositeTransform
{
    Q_OBJECT
public:
    Q_PROPERTY(int activeFrames READ get_activeFrames WRITE set_activeFrames RESET reset_activeFrames)
    BR_PROPERTY(int, activeFrames, 100)

    friend class StreamTransfrom;

    void train(const TemplateList & data)
    {
        if (!trainable) {
            qWarning("Attempted to train untrainable transform, nothing will happen.");
            return;
        }
        qFatal("Stream train is currently not implemented.");
        foreach(Transform * transform, transforms) {
            transform->train(data);
        }
    }

    bool timeVarying() const { return true; }

    void project(const Template &src, Template &dst) const
    {
        (void) src; (void) dst;
        qFatal("nope");
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        (void) src; (void) dst;
        qFatal("whatever");
    }

    // start processing, consider all templates in src a continuous
    // 'video'
    void projectUpdate(const TemplateList & src, TemplateList & dst)
    {
        dst = src;

        bool res = readStage->dataSource.open(dst);
        if (!res) return;

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
        bool wait_res = false;
        wait_res = readStage->dataSource.waitLast();

        // Now that there are no more incoming frames, call finalize
        // on each transform in turn to collect any last templates
        // they wish to issue.
        TemplateList final_output;

        // Push finalize through the stages
        for (int i=0; i < this->transforms.size(); i++)
        {
            TemplateList output_set;
            transforms[i]->finalize(output_set);

            for (int j=i+1; j < transforms.size();j++)
            {
                transforms[j]->projectUpdate(output_set);
            }
            final_output.append(output_set);
        }

        // dst is set to all output received by the final stage, along
        // with anything output via the calls to finalize.
        dst = collectionStage->getOutput();
        dst.append(final_output);

        foreach(ProcessingStage * stage, processingStages) {
            stage->reset();
        }
    }

    virtual void finalize(TemplateList & output)
    {
        (void) output;
        // Nothing in particular to do here, stream calls finalize
        // on all child transforms as part of projectUpdate
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
            it = pools.insert(this->parent(), new QThreadPool(this));
            it.value()->setMaxThreadCount(Globals->parallelism);
        }
        else it = pools.find(this->parent());
        threads = it.value();
        poolLock.unlock();

        // Are our children time varying or not? This decides whether
        // we run them in single threaded or multi threaded stages
        stage_variance.reserve(transforms.size());
        foreach (const br::Transform *transform, transforms) {
            stage_variance.append(transform->timeVarying());
        }

        // Additionally, we have a separate stage responsible for reading
        // frames from the data source
        readStage = new FirstStage(activeFrames);

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
        collectionStage = new LastStage(prev_stage_variance);
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

    ~DirectStreamTransform()
    {
        // Delete all the stages
        for (int i = 0; i < processingStages.size(); i++) {
            delete processingStages[i];
        }
        processingStages.clear();
    }

protected:
    QList<bool> stage_variance;

    FirstStage * readStage;
    LastStage * collectionStage;

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

;

class StreamTransform : public TimeVaryingTransform
{
    Q_OBJECT

public:
    StreamTransform() : TimeVaryingTransform(false)
    {
    }

    Q_PROPERTY(br::Transform *transform READ get_transform WRITE set_transform RESET reset_transform STORED false)
    Q_PROPERTY(int activeFrames READ get_activeFrames WRITE set_activeFrames RESET reset_activeFrames)
    BR_PROPERTY(br::Transform *, transform, NULL)
    BR_PROPERTY(int, activeFrames, 100)

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


    void train(const TemplateList & data)
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

        // Set up timeInvariantAlias
        // this is only safe because copies are actually made in project
        // calls, not during init.
        TimeVaryingTransform::init();

        trainable = transform->trainable;

        basis.setParent(this->parent());
        basis.transforms.clear();
        basis.activeFrames = this->activeFrames;

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

    Transform * smartCopy()
    {
        // We just want the DirectStream to begin with, so just return a copy of that.
        DirectStreamTransform * res = (DirectStreamTransform *) basis.smartCopy();
        res->activeFrames = this->activeFrames;
        return res;
    }


private:
    DirectStreamTransform basis;
};

BR_REGISTER(Transform, StreamTransform)



} // namespace br

#include "stream.moc"

