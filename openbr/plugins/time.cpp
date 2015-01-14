#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Gives time elapsed over a specified transform as a function of both images (or frames) and pixels.
 * \author Jordan Cheney \cite JordanCheney
 * \author Josh Klontz \cite jklontz
 */
class StopWatchTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description)
    BR_PROPERTY(QString, description, "Identity")

    br::Transform *transform;
    mutable QMutex mutex;
    mutable long miliseconds;
    mutable long images;
    mutable long pixels;

public:
    StopWatchTransform()
    {
        reset();
    }

private:
    void reset()
    {
        miliseconds = 0;
        images = 0;
        pixels = 0;
    }

    void init()
    {
        transform = Transform::make(description);
    }

    void train(const QList<TemplateList> &data)
    {
        transform->train(data);
    }

    void project(const Template &src, Template &dst) const
    {
        QTime watch;
        watch.start();
        transform->project(src, dst);

        QMutexLocker locker(&mutex);
        miliseconds += watch.elapsed();
        images++;
        foreach (const cv::Mat &m, src)
            pixels += (m.rows * m.cols);
    }

    void finalize(TemplateList &)
    {
        qDebug("\nProfile for \"%s\"\n"
               "\tSeconds: %g\n"
               "\tTemplates/s: %g\n"
               "\tPixels/s: %g\n",
               qPrintable(description),
               miliseconds / 1000.0,
               images * 1000.0 / miliseconds,
               pixels * 1000.0 / miliseconds);
        reset();
    }

    void store(QDataStream &stream) const
    {
        transform->store(stream);
    }

    void load(QDataStream &stream)
    {
        transform->load(stream);
    }
};

BR_REGISTER(Transform, StopWatchTransform)

} //namespace br

#include "time.moc"
