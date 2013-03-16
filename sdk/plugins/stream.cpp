#define QT_NO_DEBUG_OUTPUT

#include <openbr_plugin.h>
#include <QReadWriteLock>
#include <QWaitCondition>
#include <QThreadPool>

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
    SharedBuffer(int _maxItems = 200) : maxItems(_maxItems) {}
    virtual ~SharedBuffer() {}

    virtual void addItem(FrameData * input)=0;

    virtual FrameData * getItem()=0;

    int getMaxItems() { return maxItems; }

    virtual void stoppedInput() =0;
    virtual void startInput() = 0;
protected:
    int maxItems;
};

// For 1 - 1 boundaries, a buffer class with a single shared buffer, a mutex
// is used to serialize all access to the buffer.
class SingleBuffer : public SharedBuffer
{
public:
    SingleBuffer(unsigned _maxItems = 20) : SharedBuffer(_maxItems) { no_input = false; }

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

        // If the buffer is too full, wait for space to become available
        if (buffer.size() >= maxItems) {
            availableOutputSpace.wait(&bufferGuard);
        }

        buffer.append(input);

        // Wait for certain # of items?
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
        if (buffer.size() < maxItems / 2)
            availableOutputSpace.wakeAll();
        return output;
    }

private:
    QMutex bufferGuard;
    QWaitCondition availableInput;
    QWaitCondition availableOutputSpace;
    bool no_input;

    QList<FrameData *> buffer;
};

// Interface for sequentially getting data from some data source.
// Initialized off of a template, can represent a video file (stored in the template's filename)
// or a set of images already loaded into memory stored as multiple matrices in an input template.
class DataSource
{
public:
    DataSource() {}
    virtual ~DataSource() {}

    virtual FrameData * getNext() = 0;
    virtual void close() = 0;
    virtual bool open(Template & input) = 0;
    virtual bool isOpen() = 0;
};

// Read a video frame by frame using cv::VideoCapture
class VideoDataSource : public DataSource
{
public:
    VideoDataSource() {}

    FrameData * getNext()
    {
        if (!isOpen())
            return NULL;

        FrameData * output = new FrameData();
        output->data.append(Template(basis.file));
        output->data.last().append(cv::Mat());

        output->sequenceNumber = next_idx;
        next_idx++;

        bool res = video.read(output->data.last().last());
        if (!res) {
            delete output;
            return NULL;
        }
        return output;
    }

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
    cv::VideoCapture video;
    Template basis;
    int next_idx;
};

// Given a template as input, return its matrices one by one on subsequent calls
// to getNext
class TemplateDataSource : public DataSource
{
public:
    TemplateDataSource() { current_idx = INT_MAX; }

    FrameData * getNext()
    {
        if (!isOpen())
            return NULL;

        FrameData * output = new FrameData();
        output->data.append(basis[current_idx]);
        current_idx++;

        output->sequenceNumber = next_sequence;
        next_sequence++;

        return output;
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

    FrameData * getNext()
    {
        if (!isOpen()) return NULL;
        return actualSource->getNext();
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
            actualSource = new VideoDataSource();
            open_res = actualSource->open(input);
            qDebug("created video resource status %d", open_res);
        }
        else {
            // create frame dealer
            actualSource = new TemplateDataSource();
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
};

class ProcessingStage : public QRunnable
{
    friend class StreamTransform;
public:
    ProcessingStage()
    {
        setAutoDelete(false);
    }

    void markStart()
    {
        QMutexLocker lock(&stoppedGuard);
        stopped = false;
    }

    void waitStop()
    {
        stoppedGuard.lock();
        while (!stopped)
        {
            waitStopped.wait(&stoppedGuard);
        }
        stoppedGuard.unlock();
    }

protected:
    void markStop()
    {
        QMutexLocker lock(&stoppedGuard);
        stopped = true;
        this->waitStopped.wakeAll();
    }
    QMutex stoppedGuard;
    QWaitCondition waitStopped;
    bool stopped;

    SharedBuffer * inputBuffer;
    SharedBuffer * outputBuffer;
    Transform * transform;
    int stage_id;

public:
    // We should start, and enter a wait on input data
    void run()
    {
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
            FrameData * aFrame = dataSource.getNext();
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
            delete frame;
        }
        this->markStop();
    }
};

class StreamTransform : public CompositeTransform
{
    Q_OBJECT
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
            processingStages[i]->markStart();
            processingThreads.start(processingStages[i]);
        }

        // Start the final stage
        collectionStage.markStart();
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
        // Set up the thread pool, 1 stage for each transform, as well as first
        // and last stages, but the first stage is operated by the thread that
        // calls project so the pool only needs nTransforms+1 total.
        processingThreads.setMaxThreadCount(transforms.size() + 1);

        stage_variance.reserve(transforms.size());
        foreach (const br::Transform *transform, transforms) {
            stage_variance.append(transform->timeVarying());
        }

        // buffer 0 -- output buffer for the read stage
        sharedBuffers.append(new SingleBuffer());
        readStage.outputBuffer = sharedBuffers.last();
        readStage.stage_id = 0;

        int next_stage_id = 1;

        int lastBufferIdx = 0;
        for (int i =0; i < transforms.size(); i++)
        {
            // Set up this stage
            processingStages.append(new ProcessingStage());

            processingStages.last()->stage_id = next_stage_id++;
            processingStages.last()->inputBuffer = sharedBuffers[lastBufferIdx];
            lastBufferIdx++;

            sharedBuffers.append(new SingleBuffer());
            processingStages.last()->outputBuffer = sharedBuffers.last();
            processingStages.last()->transform = transforms[i];
        }

        collectionStage.inputBuffer = sharedBuffers.last();
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

