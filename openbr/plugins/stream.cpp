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
            qFatal("mismatched targets!");
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
    DataSource(int maxFrames=500)
    {
        final_frame = -1;
        last_issued = -2;
        last_received = -3;
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

        // The datasource broke.
        if (!res) {
            allFrames.addItem(aFrame);

            QMutexLocker lock(&last_frame_update);
            // Did we already receive the last frame?
            final_frame = last_issued;

            // We got the last frame before the data source broke,
            // better pulse lastReturned
            if (final_frame == last_received) {
                lastReturned.wakeAll();
            }
            else if (final_frame < last_received)
                std::cout << "Bad last frame " << final_frame << " but received " << last_received << std::endl;

            return NULL;
        }
        last_issued = aFrame->sequenceNumber;
        return aFrame;
    }

    // Returns true if the frame returned was the last
    // frame issued, false otherwise
    bool returnFrame(FrameData * inputFrame)
    {
        allFrames.addItem(inputFrame);

        bool rval = false;

        QMutexLocker lock(&last_frame_update);
        last_received = inputFrame->sequenceNumber;

        if (inputFrame->sequenceNumber == final_frame) {
            // We just received the last frame, better pulse
            lastReturned.wakeAll();
            rval = true;
        }

        return rval;
    }

    void waitLast()
    {
        QMutexLocker lock(&last_frame_update);
        lastReturned.wait(&last_frame_update);
    }

    virtual void close() = 0;
    virtual bool open(Template & output, int start_index=0) = 0;
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

    bool open(Template &input, int start_index=0)
    {
        final_frame = -1;
        last_issued = -2;
        last_received = -3;

        next_idx = start_index;
        basis = input;
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
        }
        else video.open(input.file.name.toStdString());

        return video.isOpened();
    }

    bool isOpen() { return video.isOpened(); }

    void close() {
        video.release();
    }

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
        output.data.last().last() = output.data.last().last().clone();

        if (!res) {
            close();
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
        data_ok = false;
    }
    bool data_ok;

    bool open(Template &input, int start_index=0)
    {
        basis = input;
        current_idx = 0;
        next_sequence = start_index;
        final_frame = -1;
        last_issued = -2;
        last_received = -3;

        data_ok = current_idx < basis.size();
        return data_ok;
    }

    bool isOpen() {
        return data_ok;
    }

    void close()
    {
        current_idx = INT_MAX;
        basis.clear();
    }

private:
    bool getNext(FrameData & output)
    {
        data_ok = current_idx < basis.size();
        if (!data_ok)
            return false;

        output.data.append(basis[current_idx]);
        current_idx++;

        output.sequenceNumber = next_sequence;
        next_sequence++;

        output.data.last().file.set("FrameNumber", output.sequenceNumber);
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

    bool open(TemplateList & input)
    {
        currentIdx = 0;
        templates = input;

        return open(templates[currentIdx]);
    }

    bool open(Template & input, int start_index=0)
    {
        close();
        final_frame = -1;
        last_issued = -2;
        last_received = -3;
        next_frame = start_index;

        // Input has no matrices? Its probably a video that hasn't been loaded yet
        if (input.empty()) {
            actualSource = new VideoDataSource(0);
            actualSource->open(input, next_frame);
        }
        else {
            // create frame dealer
            actualSource = new TemplateDataSource(0);
            actualSource->open(input, next_frame);
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
    int currentIdx;
    int next_frame;
    TemplateList templates;
    DataSource * actualSource;
    bool getNext(FrameData & output)
    {
        bool res = actualSource->getNext(output);
        if (res) {
            next_frame = output.sequenceNumber+1;
            return true;
        }

        while(!res) {
            currentIdx++;

            if (currentIdx >= templates.size())
                return false;
            bool open_res = open(templates[currentIdx], next_frame);
            if (!open_res)
                return false;
            res = actualSource->getNext(output);
        }

        next_frame = output.sequenceNumber+1;
        return res;
    }

};

class ProcessingStage;

class BasicLoop : public QRunnable
{
public:
    void run();

    QList<ProcessingStage *> * stages;
    int start_idx;
    FrameData * startItem;
};

class ProcessingStage
{
public:
    friend class StreamTransform;
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

protected:
    int thread_count;

    SharedBuffer * inputBuffer;
    ProcessingStage * nextStage;
    QList<ProcessingStage *> * stages;
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
}

class MultiThreadStage : public ProcessingStage
{
public:
    MultiThreadStage(int _input) : ProcessingStage(_input) {}


    FrameData * run(FrameData * input, bool & should_continue)
    {
        if (input == NULL) {
            qFatal("null input to multi-thread stage");
        }
        // Project the input we got
        transform->projectUpdate(input->data);

        should_continue = nextStage->tryAcquireNextStage(input);

        return input;
    }

    // Called from a different thread than run
    virtual bool tryAcquireNextStage(FrameData *& input)
    {
        (void) input;
        return true;
    }

    void reset()
    {
        // nothing to do.
    }
};


class SingleThreadStage : public ProcessingStage
{
public:
    SingleThreadStage(bool input_variance) : ProcessingStage(1)
    {
        currentStatus = STOPPING;
        next_target = 0;
        if (input_variance) {
            this->inputBuffer = new DoubleBuffer();
        }
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
        {
            BasicLoop * next = new BasicLoop();
            next->stages = stages;
            next->start_idx = this->stage_id;
            next->startItem = newItem;

            QThreadPool::globalInstance()->start(next, stages->size() - this->stage_id);
        }

        return input;
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
};

// No input buffer, instead we draw templates from some data source
// Will be operated by the main thread for the stream
class FirstStage : public SingleThreadStage
{
public:
    FirstStage() : SingleThreadStage(true) {}

    DataSourceManager dataSource;

    FrameData * run(FrameData * input, bool & should_continue)
    {
        // Is there anything on our input buffer? If so we should start a thread with that.
        QWriteLocker lock(&statusLock);
        input = dataSource.tryGetFrame();
        // Datasource broke?
        if (!input)
        {
            currentStatus = STOPPING;
            should_continue = false;
            return NULL;
        }
        lock.unlock();

        should_continue = nextStage->tryAcquireNextStage(input);

        BasicLoop * next = new BasicLoop();
        next->stages = stages;
        next->start_idx = this->stage_id;
        next->startItem = NULL;

        QThreadPool::globalInstance()->start(next, stages->size() - this->stage_id);

        return input;
    }

    // Calledfrom a different thread than run.
    bool tryAcquireNextStage(FrameData *& input)
    {
        bool was_last = dataSource.returnFrame(input);
        input = NULL;
        if (was_last) {
            return false;
        }

        if (!dataSource.isOpen())
            return false;

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
        // Ok we'll start a thread
        currentStatus = STARTING;

        // We always start a readstage thread with null input, so nothing to do here
        return true;
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

        collectedOutput.append(input->data);

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
        {
            BasicLoop * next = new BasicLoop();
            next->stages = stages;
            next->start_idx = this->stage_id;
            next->startItem = newItem;

            QThreadPool::globalInstance()->start(next, stages->size() - this->stage_id);
        }

        return input;
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

    void projectUpdate(const Template &src, Template &dst)
    {
        (void) src; (void) dst;
        qFatal("whatever");
    }

    // start processing
    void projectUpdate(const TemplateList & src, TemplateList & dst)
    {
        dst = src;

        bool res = readStage->dataSource.open(dst);
        if (!res) return;

        QThreadPool::globalInstance()->releaseThread();
        readStage->currentStatus = SingleThreadStage::STARTING;

        BasicLoop loop;
        loop.stages = &this->processingStages;
        loop.start_idx = 0;
        loop.startItem = NULL;
        loop.setAutoDelete(false);

        QThreadPool::globalInstance()->start(&loop, processingStages.size() - processingStages[0]->stage_id);

        // Wait for the end.
        readStage->dataSource.waitLast();
        QThreadPool::globalInstance()->reserveThread();

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

        // dst is set to all output received by the final stage
        dst = collectionStage->getOutput();
        dst.append(final_output);

        foreach(ProcessingStage * stage, processingStages) {
            stage->reset();
        }
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

        readStage = new FirstStage();

        processingStages.push_back(readStage);
        readStage->stage_id = 0;
        readStage->stages = &this->processingStages;

        int next_stage_id = 1;

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

            // link nextStage pointers, the stage we just appeneded is i+1 since
            // the read stage was added before this loop
            processingStages[i]->nextStage = processingStages[i+1];

            processingStages.last()->stages = &this->processingStages;

            processingStages.last()->transform = transforms[i];
            prev_stage_variance = stage_variance[i];
        }

        collectionStage = new LastStage(prev_stage_variance);
        processingStages.append(collectionStage);
        collectionStage->stage_id = next_stage_id;
        collectionStage->stages = &this->processingStages;

        processingStages[processingStages.size() - 2]->nextStage = collectionStage;

        // It's a ring buffer, get it?
        collectionStage->nextStage = readStage;
    }

    ~StreamTransform()
    {
        for (int i = 0; i < processingStages.size(); i++) {
            delete processingStages[i];
        }
    }

protected:
    QList<bool> stage_variance;

    FirstStage * readStage;
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

