#include <openbr/plugins/openbr_internal.h>

namespace br
{

class StopWatchTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(br::Transform* child READ get_child WRITE set_child RESET reset_child)
    BR_PROPERTY(br::Transform*, child, NULL)

    mutable QMutex watchLock;
    mutable long timeElapsed;
    mutable long numImgs;
    mutable long numPixels;

    ~StopWatchTransform() {
        printf("Profiled %lu images:\n"
               "\tavg time per image: %f ms\n"
               "\tavg time per pixel: %f ms\n",
               numImgs, (double) timeElapsed / numImgs, (double) timeElapsed / numPixels);
    }

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

        timeElapsed += time;
        numImgs++;
        numPixels += (src.m().rows * src.m().cols);
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

            timeElapsed += time;
            numImgs++;
            numPixels += (t.m().rows * t.m().cols);

            dst << u;
        }
    }
};
BR_REGISTER(Transform, StopWatchTransform)

} //namespace br

#include "time.moc"
