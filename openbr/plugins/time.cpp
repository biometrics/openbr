#include <openbr/plugins/openbr_internal.h>

namespace br
{

class StopWatchTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(br::Transform* child READ get_child WRITE set_child RESET reset_child)
    BR_PROPERTY(br::Transform*, child, NULL)

    mutable QMutex watchLock;

    void project(const Template &src, Template &dst) const
    {
        QTime watch;

        if (child == NULL)
            qFatal("Can't find child transform! Command line syntax is StopWatch(name of transform to profile)");

        watchLock.lock();
        watch.start();

        child->project(src, dst);

        int time = watch.elapsed();
        watchLock.unlock();

        dst.file.set("time", QVariant::fromValue(time));
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        QTime watch;
        foreach (const Template &t, src) {
            watchLock.lock();
            watch.start();

            Template u;
            child->project(t, u);

            int time = watch.elapsed();
            watchLock.unlock();

            u.file.set("time", QVariant::fromValue(time));
            dst << u;
        }
    }
};
BR_REGISTER(Transform, StopWatchTransform)

class StopWatchProfiler : public TimeVaryingTransform
{
    Q_OBJECT

    TemplateList buffer;

public:
    StopWatchProfiler() : TimeVaryingTransform(false, false) {}

private:
    void train(const TemplateList &data)
    {
       (void) data;
    }
    void projectUpdate(const Template &src, Template &dst)
    {
        dst = src;
        buffer.append(src);
    } 
    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst = src;
        buffer.append(src);
    }

    void finalize(TemplateList &output) {
    printf("\n\nProfiling data....\n\n");

	output = buffer;

        if (buffer.isEmpty())
            qFatal("Empty buffer! Something must have gone wrong.");
        
        if (!buffer.first().file.contains("time"))
            qFatal("No time attribute in the metadata! Did you forget your StopWatch?");

        unsigned long imgs = buffer.length();
        unsigned long totalTime = 0;

        foreach(const Template &t, buffer) {
            totalTime += t.file.value("time").toUInt();
        }

        printf("Profiled %lu images:\n"
               "\tavg time per image: %f ms\n",
               imgs, (double)totalTime / imgs);
    }
};
BR_REGISTER(Transform, StopWatchProfiler)

} //namespace br

#include "time.moc"
