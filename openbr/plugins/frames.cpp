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

    void train(const TemplateList &data)
    {
        (void) data;
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        buffer.append(src);
        if (buffer.size() < n) return;
        foreach (const Template &t, buffer) dst.append(t);
        dst.file = buffer.takeFirst().file;
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

#include "frames.moc"
