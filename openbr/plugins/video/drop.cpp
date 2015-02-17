#include <openbr/plugins/openbr_internal.h>

namespace br
{

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

#include "video/drop.moc"
