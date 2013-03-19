#include <QReadWriteLock>
#include <QWaitCondition>
#include <QThreadPool>
#include <QSemaphore>
#include <QMap>
#include <opencv/highgui.h>
#include <QtConcurrent>
#include <openbr/openbr_plugin.h>

#include "openbr/core/common.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/qtutils.h"

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

    virtual FrameData * tryGetItem()=0;
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
            qWarning("mismatched targets!");
        }

        next_target = next_target + 1;

        FrameData * output = result.value();
        buffer.erase(result);
        return output;
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
    DataSource(int maxFrames=Globals->parallelism + 1)
    {
        final_frame = -1;
        last_issued = -2;
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
    FrameData * tryGetFrame()
    {
        FrameData * aFrame = allFrames.tryGetItem();
        if (aFrame == NULL)
            return NULL;

        aFrame->data.clear();
        aFrame->sequenceNumber = -1;

        bool res = getNext(*aFrame);
        if (!res) {
            allFrames.addItem(aFrame);
            // Datasource broke?
            QMutexLocker lock(&last_frame_update);

            final_frame = last_issued;
            if (final_frame == last_received)
                lastReturned.wakeAll();
            else if (final_frame < last_received)
                std::cout << "Bad last frame " << final_frame << " but received " << last_received << std::endl;
            return NULL;
        }
        last_issued = aFrame->sequenceNumber;
        return aFrame;
    }

    bool returnFrame(FrameData * inputFrame)
    {
        allFrames.addItem(inputFrame);

        QMutexLocker lock(&last_frame_update);
        last_received = inputFrame->sequenceNumber;
        if (inputFrame->sequenceNumber == final_frame) {
            lastReturned.wakeAll();
        }

        return this->final_frame != -1;
    }

    void waitLast()
    {
        QMutexLocker lock(&last_frame_update);
        lastReturned.wait(&last_frame_update);
    }

    virtual void close() = 0;
    virtual bool open(Template & output) = 0;
    virtual bool isOpen() = 0;

    virtual bool getNext(FrameData & input) = 0;

protected:
    DoubleBuffer allFrames;
    int final_frame;
    int last_issued;
    int last_received;

    QWaitCondition lastReturned;
    QMutex last_frame_update;
};

// Read a video frame by frame using cv::VideoCapture
class VideoDataSource : public DataSource
{
public:
    VideoDataSource(int maxFrames) : DataSource(maxFrames) {}

    bool open(Template &input)
    {
        final_frame = -1;
        last_issued = -2;

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
        output.data.last().file.set("FrameNumber", output.sequenceNumber);
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
        final_frame = -1;
        last_issued = -2;

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
        final_frame = -1;
        last_issued = -2;

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
public:
    friend class StreamTransform;
public:
    ProcessingStage(int nThreads = 1)
    {
        thread_count = nThreads;
        setAutoDelete(false);
    }

    virtual void run()=0;

    virtual void nextStageRun(FrameData * input)=0;

protected:
    int thread_count;

    SharedBuffer * inputBuffer;
    ProcessingStage * nextStage;
    Transform * transform;
    int stage_id;

};

class MultiThreadStage;

void multistage_run(MultiThreadStage * basis, FrameData * input);

class MultiThreadStage : public ProcessingStage
{
public:
    MultiThreadStage(int _input) : ProcessingStage(_input) {}

    friend void multistage_run(MultiThreadStage * basis, FrameData * input);

    void run()
    {
        qFatal("no don't do it!");
    }

    // Called from a different thread than run
    virtual void nextStageRun(FrameData * input)
    {
        QtConcurrent::run(multistage_run, this, input);
    }
};

void multistage_run(MultiThreadStage * basis, FrameData * input)
{
    if (input == NULL)
        qFatal("null input to multi-thread stage");
    // Project the input we got
    basis->transform->projectUpdate(input->data);

    basis->nextStage->nextStageRun(input);
}

class SingleThreadStage : public ProcessingStage
{
public:
    SingleThreadStage(bool input_variance) : ProcessingStage(1)
    {
        currentStatus = STOPPING;
        next_target = 0;
        if (input_variance)
            this->inputBuffer = new DoubleBuffer();
        else
            this->inputBuffer = new SequencingBuffer();
    }
    ~SingleThreadStage()
    {
        delete inputBuffer;
    }

    int next_target;
    enum Status
    {
        STARTING,
        STOPPING
    };
    QReadWriteLock statusLock;
    Status currentStatus;

    // We should start, and enter a wait on input data
    void run()
    {
        FrameData * currentItem;
        forever
        {
            // Whether or not we get a valid item controls whether or not we
            QWriteLocker lock(&statusLock);
            currentItem = inputBuffer->tryGetItem();
            if (currentItem == NULL)
            {
                this->currentStatus = STOPPING;
                return;
            }
            lock.unlock();
            if (currentItem->sequenceNumber != next_target)
            {
                qFatal("out of order frames for stage %d, got %d expected %d", this->stage_id, currentItem->sequenceNumber, this->next_target);
            }
            next_target = currentItem->sequenceNumber + 1;

            // Project the input we got
            transform->projectUpdate(currentItem->data);

            this->nextStage->nextStageRun(currentItem);
        }
    }

    // Calledfrom a different thread than run.
    void nextStageRun(FrameData * input)
    {
        // add to our input buffer
        inputBuffer->addItem(input);
        QReadLocker lock(&statusLock);
        if (currentStatus == STARTING)
            return;

        // Have to change to a write lock to modify currentStatus
        lock.unlock();
        QWriteLocker writeLock(&statusLock);
        // But someone might have changed it between locks
        if (currentStatus == STARTING)
            return;
        // Ok we can start a thread
        QThreadPool::globalInstance()->start(this);
        currentStatus = STARTING;
    }
};

// No input buffer, instead we draw templates from some data source
// Will be operated by the main thread for the stream
class FirstStage : public SingleThreadStage
{
public:
    FirstStage() : SingleThreadStage(true) {}

    DataSourceManager dataSource;
    // Start drawing frames from the datasource.
    void run()
    {
        FrameData * currentItem;
        forever
        {
            // Whether or not we get a valid item controls whether or not we
            QWriteLocker lock(&statusLock);

            currentItem = this->dataSource.tryGetFrame();
            if (currentItem == NULL)
            {
                this->currentStatus = STOPPING;
                return;
            }
            lock.unlock();
            if (currentItem->sequenceNumber != next_target)
            {
                qFatal("out of order frames for stage %d, got %d expected %d", this->stage_id, currentItem->sequenceNumber, this->next_target);
            }
            next_target = currentItem->sequenceNumber + 1;

            this->nextStage->nextStageRun(currentItem);
        }
    }

    void nextStageRun(FrameData * input)
    {
        QWriteLocker lock(&statusLock);

        // Return the frame to the frame buffer
        bool res = dataSource.returnFrame(input);
        // If the data source broke already, we're done.
        if (res)
            return;

        if (currentStatus == STARTING)
            return;

        currentStatus = STARTING;
        QThreadPool::globalInstance()->start(this, this->next_target);
    }

};

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
    void run()
    {
        forever
        {
            QWriteLocker lock(&statusLock);
            FrameData * currentItem = inputBuffer->tryGetItem();
            if (currentItem == NULL)
            {
                currentStatus = STOPPING;
                break;
            }
            lock.unlock();

            if (currentItem->sequenceNumber != next_target)
            {
                qFatal("out of order frames for collection stage %d, got %d expected %d", this->stage_id, currentItem->sequenceNumber, this->next_target);
            }
            next_target = currentItem->sequenceNumber + 1;

            // Just put the item on collectedOutput
            collectedOutput.append(currentItem->data);
            this->nextStage->nextStageRun(currentItem);
        }
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

        QThreadPool::globalInstance()->releaseThread();
        readStage.currentStatus = SingleThreadStage::STARTING;
        QThreadPool::globalInstance()->start(&readStage, 0);
        // Wait for the end.
        readStage.dataSource.waitLast();
        QThreadPool::globalInstance()->reserveThread();

        // dst is set to all output received by the final stage
        dst = collectionStage->getOutput();
    }

    virtual void finalize(TemplateList & output)
    {
        (void) output;
        // Not handling this yet -cao
    }

    // Create and link stages
    void init()
    {
        if (transforms.isEmpty()) return;

        stage_variance.reserve(transforms.size());
        foreach (const br::Transform *transform, transforms) {
            stage_variance.append(transform->timeVarying());
        }

        readStage.stage_id = 0;

        int next_stage_id = 1;
        int lastBufferIdx = 0;
        bool prev_stage_variance = true;
        for (int i =0; i < transforms.size(); i++)
        {
            if (stage_variance[i])
            {
                processingStages.append(new SingleThreadStage(prev_stage_variance));
            }
            else
                processingStages.append(new MultiThreadStage(Globals->parallelism));

            processingStages.last()->stage_id = next_stage_id++;

            // link nextStage pointers
            if (i == 0)
                this->readStage.nextStage = processingStages[i];
            else
                processingStages[i-1]->nextStage = processingStages[i];

            lastBufferIdx++;

            processingStages.last()->transform = transforms[i];
            prev_stage_variance = stage_variance[i];
        }

        collectionStage = new LastStage(prev_stage_variance);
        collectionStage->stage_id = next_stage_id;

        // It's a ring buffer, get it?
        processingStages.last()->nextStage = collectionStage;
        collectionStage->nextStage = &readStage;
    }

    ~StreamTransform()
    {
        for (int i = 0; i < processingStages.size(); i++) {
            delete processingStages[i];
        }
        delete collectionStage;
    }

protected:
    QList<bool> stage_variance;

    FirstStage readStage;
    LastStage * collectionStage;

    QList<ProcessingStage *> processingStages;

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

