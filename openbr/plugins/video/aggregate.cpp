#include <openbr/plugins/openbr_internal.h>

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

    void finalize(TemplateList &output)
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
};

BR_REGISTER(Transform, AggregateFrames)

} // namespace br

#include "video/aggregate.moc"
