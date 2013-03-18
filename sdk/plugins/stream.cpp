#include <openbr_plugin.h>
#include <QReadWriteLock>
#include <QWaitCondition>
#include <QThreadPool>
#include <QSemaphore>
#include <QMap>

#include "core/common.h"
#include "core/opencvutils.h"
#include "core/qtutils.h"

#include "opencv/highgui.h"

#include <iostream>

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

    virtual FrameData * getItem()=0;

    virtual void stoppedInput() =0;
    virtual void startInput() = 0;
};

// For 1 - n boundaries, a buffer class with a single shared buffer, a mutex
// is used to serialize all access to the buffer.
class SingleBuffer : public SharedBuffer
{
public:
    SingleBuffer() { no_input = false; }

    void stoppedInput()
    {
        QMutexLocker bufferLock(&bufferGuard);
        no_input = true;
        // Release anything waiting for input items.
        availableInput.wakeAll();
    }

    // There will be more input
    void startInput()
    {
        QMutexLocker bufferLock(&bufferGuard);
        no_input = false;
    }

    void addItem(FrameData * input)
    {
        QMutexLocker bufferLock(&bufferGuard);

        buffer.append(input);

        availableInput.wakeOne();
    }

    FrameData * getItem()
    {
        QMutexLocker bufferLock(&bufferGuard);

        if (buffer.empty()) {
            // If no further items will come we are done here
            if (no_input)
                return NULL;
            // Wait for an item
            availableInput.wait(&bufferGuard);
        }

        // availableInput was signalled, but the buffer is still empty? We're done here.
        if (buffer.empty())
            return NULL;

        FrameData * output = buffer.first();
        buffer.removeFirst();
        return output;
    }

private:
    QMutex bufferGuard;
    QWaitCondition availableInput;
    bool no_input;

    QList<FrameData *> buffer;
};

// for n - 1 boundaries, multiple threads call addItem, the frames are
// sequenced based on FrameData::sequence_number, and calls to getItem
// receive them in that order
class SequencingBuffer : public SharedBuffer
{
public:
    SequencingBuffer()
    {
        no_input = false;
        next_target = 0;
    }

    void stoppedInput()
    {
        QMutexLocker bufferLock(&bufferGuard);
        no_input = true;
        // Release anything waiting for input items.
        availableInput.wakeAll();
    }

    // There will be more input
    void startInput()
    {
        QMutexLocker bufferLock(&bufferGuard);
        no_input = false;
    }

    void addItem(FrameData * input)
    {
        QMutexLocker bufferLock(&bufferGuard);

        buffer.insert(input->sequenceNumber, input);

        if (input->sequenceNumber == next_target) {
            availableInput.wakeOne();
        }
    }

    FrameData * getItem()
    {
        QMutexLocker bufferLock(&bufferGuard);

        if (buffer.empty() || buffer.begin().key() != this->next_target) {
            if (buffer.empty() && no_input) {
                next_target = 0;
                return NULL;
            }
            availableInput.wait(&bufferGuard);
        }

        // availableInput was signalled, but the buffer is empty? We're done here.
        if (buffer.empty()) {
            next_target = 0;
            return NULL;
        }

        QMap<int, FrameData *>::Iterator result = buffer.begin();
        //next_target++;
        if (next_target != result.value()->sequenceNumber) {
            qWarning("mismatched targets!");
        }

        next_target = next_target + 1;

        FrameData * output = result.value();
        buffer.erase(result);
        return output;
    }

private:
    QMutex bufferGuard;
    QWaitCondition availableInput;
    bool no_input;

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

    void stoppedInput()
    {
        QWriteLocker bufferLock(&bufferGuard);
        no_input = true;
        // Release anything waiting for input items.
        availableInput.wakeAll();
    }

    // There will be more input
    void startInput()
    {
        QWriteLocker bufferLock(&bufferGuard);
        no_input = false;
    }

    // called from the producer thread
    void addItem(FrameData * input)
    {
        QReadLocker readLock(&bufferGuard);
        inputBuffer->append(input);
        availableInput.wakeOne();
    }

    // Called from the consumer thread
    FrameData * getItem() {
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
            // If nothing else is coming, return null
            if (no_input)
                return NULL;
            //otherwise, wait on the input buffer
            availableInput.wait(&bufferGuard);
            // Did we get woken up because no more input is coming? if so
            // we're done here
            if (no_input && inputBuffer->empty())
                return NULL;
        }

        // input buffer is non-empty, so swap the buffers
        std::swap(inputBuffer, outputBuffer);

        // Return a frame
        FrameData * output = outputBuffer->first();
        outputBuffer->removeFirst();
        return output;
    }

private:
    // The read-write lock. The thread adding to this buffer can add
    // to the current input buffer if it has a read lock. The thread
    // removing from this buffer can remove things from the current
    // output buffer if it has a read lock, or swap the buffers if it
    // has a write lock.
    // Checking/modifying no_input requires a write lock.
    QReadWriteLock bufferGuard;
    QWaitCondition availableInput;
    bool no_input;

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
    DataSource(int maxFrames=100)
    {
        for (int i=0; i < maxFrames;i++)
        {
            allFrames.addItem(new FrameData());
        }
        allFrames.startInput();
    }

    virtual ~DataSource()
    {
        allFrames.stoppedInput();
        while (true)
        {
            FrameData * frame = allFrames.getItem();
            if (frame == NULL)
                break;
            delete frame;
        }
    }

    FrameData * getFrame()
    {
        FrameData * aFrame = allFrames.getItem();
        aFrame->data.clear();
        aFrame->sequenceNumber = -1;

        bool res = getNext(*aFrame);
        if (!res) {
            allFrames.addItem(aFrame);
            return NULL;
        }
        return aFrame;
    }

    void returnFrame(FrameData * inputFrame)
    {
        allFrames.addItem(inputFrame);
    }

    virtual void close() = 0;
    virtual bool open(Template & output) = 0;
    virtual bool isOpen() = 0;

    virtual bool getNext(FrameData & input) = 0;

protected:
    DoubleBuffer allFrames;
};

// Read a video frame by frame using cv::VideoCapture
class VideoDataSource : public DataSource
{
public:
    VideoDataSource(int maxFrames) : DataSource(maxFrames) {}

    bool open(Template &input)
    {
        next_idx = 0;
        basis = input;
        video.open(input.file.name.toStdString());
        return video.isOpened();
    }

    bool isOpen() { return video.isOpened(); }

    void close() { video.release(); }

private:
    bool getNext(FrameData & output)
    {
        if (!isOpen())
            return false;

        output.data.append(Template(basis.file));
        output.data.last().append(cv::Mat());

        output.sequenceNumber = next_idx;
        next_idx++;

        bool res = video.read(output.data.last().last());
        if (!res) {
            return false;
        }
        return true;
    }

    cv::VideoCapture video;
    Template basis;
    int next_idx;
};

// Given a template as input, return its matrices one by one on subsequent calls
// to getNext
class TemplateDataSource : public DataSource
{
public:
    TemplateDataSource(int maxFrames) : DataSource(maxFrames)
    {
        current_idx = INT_MAX;
    }

    bool open(Template &input)
    {
        basis = input;
        current_idx = 0;
        next_sequence = 0;
        return isOpen();
    }

    bool isOpen() { return current_idx < basis.size() ; }

    void close()
    {
        current_idx = INT_MAX;
        basis.clear();
    }

private:
    bool getNext(FrameData & output)
    {
        if (!isOpen())
            return false;

        output.data.append(basis[current_idx]);
        current_idx++;

        output.sequenceNumber = next_sequence;
        next_sequence++;

        return true;
    }

    Template basis;
    int current_idx;
    int next_sequence;
};

// Given a template as input, create a VideoDataSource or a TemplateDataSource
// depending on whether or not it looks like the input template has already
// loaded frames into memory.
class DataSourceManager : public DataSource
{
public:
    DataSourceManager()
    {
        actualSource = NULL;
    }

    ~DataSourceManager()
    {
        close();
    }

    void close()
    {
        if (actualSource) {
            actualSource->close();
            delete actualSource;
            actualSource = NULL;
        }
    }

    bool open(Template & input)
    {
        close();
        bool open_res = false;
        // Input has no matrices? Its probably a video that hasn't been loaded yet
        if (input.empty()) {
            actualSource = new VideoDataSource(0);
            open_res = actualSource->open(input);
        }
        else {
            // create frame dealer
            actualSource = new TemplateDataSource(0);
            open_res = actualSource->open(input);
        }
        if (!isOpen()) {
            delete actualSource;
            actualSource = NULL;
            return false;
        }
        return true;
    }

    bool isOpen() { return !actualSource ? false : actualSource->isOpen(); }

protected:
    DataSource * actualSource;
    bool getNext(FrameData & output)
    {
        return actualSource->getNext(output);
    }

};

class ProcessingStage : public QRunnable
{
    friend class StreamTransform;
public:
    ProcessingStage(int nThreads = 1)
    {
        thread_count = nThreads;
        activeThreads.release(thread_count);
        setAutoDelete(false);
    }

    void markStart()
    {
        activeThreads.acquire();
    }

    void waitStop()
    {
        // Wait until all threads have stopped
        activeThreads.acquire(thread_count);
        activeThreads.release(thread_count);
    }

protected:
    void markStop()
    {
        activeThreads.release();
    }
    QSemaphore activeThreads;
    int thread_count;

    SharedBuffer * inputBuffer;
    SharedBuffer * outputBuffer;
    Transform * transform;
    int stage_id;

public:
    // We should start, and enter a wait on input data
    void run()
    {
        markStart();
        forever
        {
            FrameData * currentItem = inputBuffer->getItem();
            if (currentItem == NULL)
                break;
            // Project the input we got
            transform->projectUpdate(currentItem->data);
            // Add the result to the ouptut buffer
            outputBuffer->addItem(currentItem);
        }
        markStop();
    }
};


// No input buffer, instead we draw templates from some data source
// Will be operated by the main thread for the stream
class FirstStage : public ProcessingStage
{
public:
    DataSourceManager dataSource;
    // Start drawing frames from the datasource.
    void run()
    {
        forever
        {
            //FrameData * aFrame = dataSource.getNext();
            FrameData * aFrame = dataSource.getFrame();
            if (aFrame == NULL)
                break;
            outputBuffer->addItem(aFrame);
        }
        this->markStop();
    }
};

class LastStage : public ProcessingStage
{
public:
    TemplateList getOutput()
    {
        return collectedOutput;
    }

private:
    TemplateList collectedOutput;
public:
    DataSource * data;
    void run()
    {
        forever
        {
            // Wait for input
            FrameData * frame = inputBuffer->getItem();
            if (frame == NULL)
                break;
            // Just put the item on collectedOutput
            collectedOutput.append(frame->data);
            // Return the frame to the input frame buffer
            data->returnFrame(frame);
        }
        this->markStop();
    }
};

class StreamTransform : public CompositeTransform
{
    Q_OBJECT
    int threads_per_multi_stage;
public:
    void train(const TemplateList & data)
    {
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
    void project(const TemplateList & src, TemplateList & dst) const
    {
        (void) src; (void) dst;
        qFatal("nope");
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        (void) src; (void) dst;
        qFatal("whatever");
    }

    // start processing
    void projectUpdate(const TemplateList & src, TemplateList & dst)
    {
        if (src.size() != 1)
            qFatal("Expected single template input to stream");

        dst = src;
        bool res = readStage.dataSource.open(dst[0]);
        if (!res) {
            qWarning("failed to stream template %s", qPrintable(dst[0].file.name));
            return;
        }

        // Tell all buffers to expect input
        for (int i=0; i < sharedBuffers.size(); i++) {
            sharedBuffers[i]->startInput();
        }

        // Start our processing stages
        for (int i=0; i < this->processingStages.size(); i++) {
            int count = stage_variance[i] ? 1 : threads_per_multi_stage;
            for (int j =0; j < count; j ++) processingThreads.start(processingStages[i]);
        }

        // Start the final stage
        processingThreads.start(&collectionStage);

        // Run the read stage ourselves
        readStage.run();

        // The read stage has stopped (since we ran the read stage).
        // Step over the buffers, and call stoppedInput to tell the stage
        // reading from each buffer that no more frames will be added after
        // the current ones run out, then wait for the thread to finish.
        for (int i =0; i < (sharedBuffers.size() - 1); i++) {
            // Indicate that no more input will be available
            sharedBuffers[i]->stoppedInput();

            // Wait for the thread to finish.
            this->processingStages[i]->waitStop();
        }
        // Wait for the collection stage to finish
        sharedBuffers.last()->stoppedInput();
        collectionStage.waitStop();

        // dst is set to all output received by the final stage
        dst = collectionStage.getOutput();
    }

    virtual void finalize(TemplateList & output)
    {
        (void) output;
        // Not handling this yet -cao
    }

    // Create and link stages
    void init()
    {
        int thread_count = 0;
        threads_per_multi_stage = 4;
        stage_variance.reserve(transforms.size());
        foreach (const br::Transform *transform, transforms) {
            stage_variance.append(transform->timeVarying());
            thread_count += transform->timeVarying() ? 1 : threads_per_multi_stage;
        }
        if (transforms.isEmpty()) return;

        // Set up the thread pool, 1 stage for each transform, as well as first
        // and last stages, but the first stage is operated by the thread that
        // calls project so the pool only needs nTransforms+1 total.
        processingThreads.setMaxThreadCount(thread_count + 1);


        // buffer 0 -- output buffer for the read stage, input buffer for
        // first transform. Is that transform time-varying?
        if (stage_variance[0])
            sharedBuffers.append(new DoubleBuffer());
        // If not, we can run multiple threads
        else
            sharedBuffers.append(new SingleBuffer());

        readStage.outputBuffer = sharedBuffers.last();
        readStage.stage_id = 0;

        int next_stage_id = 1;

        int lastBufferIdx = 0;
        for (int i =0; i < transforms.size(); i++)
        {
            // Set up this stage
            processingStages.append(new ProcessingStage(stage_variance[i] ? 1 : threads_per_multi_stage));

            processingStages.last()->stage_id = next_stage_id++;
            processingStages.last()->inputBuffer = sharedBuffers[lastBufferIdx];
            lastBufferIdx++;

            // This stage's output buffer, next stage's input buffer. If this is
            // the last transform, the next stage is the (time varying) collection
            // stage
            bool next_variance = (i+1) < transforms.size() ? stage_variance[i+1] : true;
            bool current_variance = stage_variance[i];
            // if this is a single threaded stage
            if (current_variance)
            {
                // 1 - 1 case
                if (next_variance)
                    sharedBuffers.append(new DoubleBuffer());
                // 1 - n case
                else
                    sharedBuffers.append(new SingleBuffer());
            }
            // This is a multi-threaded stage
            else
            {
                // If the next stage is single threaded, we need to sequence our
                // output (n - 1 case)
                if (next_variance)
                    sharedBuffers.append(new SequencingBuffer());
                // Otherwise, this is an n-n boundary and we don't need to
                // adhere to any particular sequence
                else
                    sharedBuffers.append(new SingleBuffer());
            }
            processingStages.last()->outputBuffer = sharedBuffers.last();
            processingStages.last()->transform = transforms[i];
        }

        collectionStage.inputBuffer = sharedBuffers.last();
        collectionStage.data = &readStage.dataSource;
        collectionStage.stage_id = next_stage_id;
    }

    ~StreamTransform()
    {
        for (int i = 0; i < processingStages.size(); i++) {
            delete processingStages[i];
        }
        for (int i = 0; i < sharedBuffers.size(); i++) {
            delete sharedBuffers[i];
        }

    }

protected:
    QList<bool> stage_variance;

    FirstStage readStage;
    LastStage collectionStage;

    QList<ProcessingStage *> processingStages;
    QList<SharedBuffer *> sharedBuffers;

    QThreadPool processingThreads;

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

BR_REGISTER(Transform, StreamTransform)


} // namespace br

#include "stream.moc"

