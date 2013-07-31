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
        if (aFrame == NULL) {
            return NULL;
        }

        aFrame->data.clear();
        aFrame->sequenceNumber = -1;

        // Try to read a frame, if this returns false the data source is broken
        bool res = getNext(*aFrame);

        // The datasource broke, update final_frame
        if (!res)
        {
            QMutexLocker lock(&last_frame_update);
            final_frame = lookAhead.back()->sequenceNumber;
            allFrames.addItem(aFrame);
        }
        else lookAhead.push_back(aFrame);

        FrameData * rVal = lookAhead.first();
        lookAhead.pop_front();


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

        allFrames.addItem(inputFrame);

        bool rval = false;

        QMutexLocker lock(&last_frame_update);

        if (frameNumber == final_frame) {
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

    bool open(Template & output, int start_index = 0)
    {
        is_broken = false;
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
        // from it.
        if (!res) {
            is_broken = true;
            return false;
        }

        lookAhead.append(firstFrame);
        return true;
    }

    virtual bool isOpen()=0;
    virtual bool concreteOpen(Template & output) = 0;
    virtual bool getNext(FrameData & input) = 0;
    virtual void close() = 0;

    int next_sequence_number;
protected:
    DoubleBuffer allFrames;
    int final_frame;
    bool is_broken;
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
            video.open(QFileInfo(input.file.name).absoluteFilePath().toStdString());
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
        if (!isOpen())
            return false;

        output.data.append(Template(basis.file));
        output.data.last().m() = cv::Mat();

        output.sequenceNumber = next_sequence_number;
        next_sequence_number++;

        cv::Mat temp;
        bool res = video.read(temp);

        if (!res) {
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
    DataSourceManager() : DataSource(500)
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
        current_template_idx = 0;
        templates = input;

        return DataSource::open(templates[current_template_idx]);
    }

    bool concreteOpen(Template & input)
    {
        close();

        // Input has no matrices? Its probably a video that hasn't been loaded yet
        if (input.empty()) {
            actualSource = new VideoDataSource(0);
            actualSource->concreteOpen(input);
        }
        else {
            // create frame dealer
            actualSource = new TemplateDataSource(0);
            actualSource->concreteOpen(input);
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
    // Index of the template in the templatelist we are currently reading from
    int current_template_idx;

    TemplateList templates;
    DataSource * actualSource;
    bool getNext(FrameData & output)
    {
        bool res = actualSource->getNext(output);
        output.sequenceNumber = next_sequence_number;

        if (res) {
            output.data.last().file.set("FrameNumber", output.sequenceNumber);
            next_sequence_number++;
            if (output.data.last().last().empty())
                qDebug("broken matrix");
            return true;
        }


        while(!res) {
            output.data.clear();
            current_template_idx++;

            // No more templates? We're done
            if (current_template_idx >= templates.size())
                return false;

            // open the next data source
            bool open_res = concreteOpen(templates[current_template_idx]);
            if (!open_res)
                return false;

            // get a frame from it
            res = actualSource->getNext(output);
        }
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
            startThread(newItem);

        return input;
    }

    void startThread(br::FrameData * newItem)
    {
        BasicLoop * next = new BasicLoop();
        next->stages = stages;
        next->start_idx = this->stage_id;
        next->startItem = newItem;
        this->threads->start(next, stages->size() - stage_id);
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
// Will be operated by the main thread for the stream. starts threads
class FirstStage : public SingleThreadStage
{
public:
    FirstStage() : SingleThreadStage(true) {}

    DataSourceManager dataSource;

    FrameData * run(FrameData * input, bool & should_continue)
    {
        // Try to get a frame from the datasource
        QWriteLocker lock(&statusLock);
        bool last_frame = false;
        input = dataSource.tryGetFrame(last_frame);

        // Datasource broke, or is currently out of frames?
        if (!input || last_frame)
        {
            // We will just stop and not continue.
            currentStatus = STOPPING;
            if (!input) {
                should_continue = false;
                return NULL;
            }
        }
        lock.unlock();
        // Can we enter the next stage?
        should_continue = nextStage->tryAcquireNextStage(input);

        // We are exiting leaving this stage, should we start another
        // thread here? Normally we will always re-queue a thread on
        // the first stage, but if we received the last frame there is
        // no need to.
        if (!last_frame) {
            startThread(NULL);
        }

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
        // A thread is already in the first stage,
        // we should just return
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

// starts threads
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

        // Start the first thread in the stream.
        readStage->currentStatus = SingleThreadStage::STARTING;
        readStage->startThread(NULL);

        // Wait for the stream to reach the last frame available from
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

        stage_variance.reserve(transforms.size());
        foreach (const br::Transform *transform, transforms) {
            stage_variance.append(transform->timeVarying());
        }

        readStage = new FirstStage();

        processingStages.push_back(readStage);
        readStage->stage_id = 0;
        readStage->stages = &this->processingStages;
        readStage->threads = this->threads;

        int next_stage_id = 1;

        bool prev_stage_variance = true;
        for (int i =0; i < transforms.size(); i++)
        {
            if (stage_variance[i])
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

        collectionStage = new LastStage(prev_stage_variance);
        processingStages.append(collectionStage);
        collectionStage->stage_id = next_stage_id;
        collectionStage->stages = &this->processingStages;
        collectionStage->threads = this->threads;

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

QHash<QObject *, QThreadPool *> StreamTransform::pools;
QMutex StreamTransform::poolsAccess;

BR_REGISTER(Transform, StreamTransform)


} // namespace br

#include "stream.moc"

