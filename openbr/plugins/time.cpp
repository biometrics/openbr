#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Gives time elapsed over a specified transform as a function of both images (or frames) and pixels.
 * \author Jordan Cheney \cite JordanCheney
 */
class StopWatchTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(br::Transform* child READ get_child WRITE set_child RESET reset_child)
    BR_PROPERTY(br::Transform*, child, NULL)

    mutable long timeElapsed;
    mutable long numImgs;
    mutable long numPixels;

public:
    StopWatchTransform() : TimeVaryingTransform(false, false)
    {
        timeElapsed = 0;
        numImgs = 0;
        numPixels = 0;
    }

private:
    void projectUpdate(const Template &src, Template &dst)
    {
        QTime watch;

        if (child == NULL)
            qFatal("Can't find child transform! Command line syntax is StopWatch(name of transform to profile)");

        watch.start();

        child->project(src, dst);

        int time = watch.elapsed();

        timeElapsed += time;
        numImgs++;
        numPixels += (src.m().rows * src.m().cols);
    }

    void finalize(TemplateList &output)
    {
        (void)output;

        printf("\n\nProfiled %lu images:\n"
        "\tavg time per image: %f ms\n"
        "\tavg time per pixel: %f ms\n",numImgs, (double)timeElapsed / numImgs, (double)timeElapsed / numPixels);

        timeElapsed = 0;
        numImgs = 0;
        numPixels = 0;
    }
};

BR_REGISTER(Transform, StopWatchTransform)

} //namespace br

#include "time.moc"
