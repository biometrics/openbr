#include "openbr_internal.h"

namespace br
{

/*!
 * \ingroup transforms
 * \brief Passes along n sequential frames to the next transform.
 * \author Josh Klontz \cite jklontz
 *
 * For a video with m frames, AggregateFrames would create a total of m-n+1 sequences ([0,n] ... [m-n+1, m]).
 */
class AggregateFrames : public TimeVaryingTransform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    BR_PROPERTY(int, n, 1)

    TemplateList buffer;

public:
    AggregateFrames() : TimeVaryingTransform(false, false) {}

private:
    void train(const TemplateList &data)
    {
        (void) data;
    }

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        buffer.append(src);
        if (buffer.size() < n) return;
        Template out;
        foreach (const Template &t, buffer) out.append(t);
        out.file = buffer.takeFirst().file;
        dst.append(out);
    }

    void finalize(TemplateList & output)
    {
        (void) output;
        buffer.clear();
    }

    void store(QDataStream &stream) const
    {
        (void) stream;
    }

    void load(QDataStream &stream)
    {
        (void) stream;
    }

    void init()
    {
        TimeVaryingTransform::init();
    }
};

BR_REGISTER(Transform, AggregateFrames)

/*!
 * \ingroup transforms
 * \brief Only use one frame every n frames.
 * \author Austin Blanton \cite imaus10
 *
 * For a video with m frames, DropFrames will pass on m/n frames.
 */
class DropFrames : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    BR_PROPERTY(int, n, 1)

    void project(const TemplateList &src, TemplateList &dst) const
    {
        if (src.first().file.get<int>("FrameNumber") % n != 0) return;
        dst = src;
    }

    void project(const Template &src, Template &dst) const
    {
        (void) src; (void) dst; qFatal("shouldn't be here");
    }
};

BR_REGISTER(Transform, DropFrames)

} // namespace br

#include "frames.moc"
