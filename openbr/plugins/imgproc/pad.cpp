#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

class PadTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(int padSize READ get_padSize WRITE set_padSize RESET reset_padSize STORED false)
    Q_PROPERTY(int padValue READ get_padValue WRITE set_padValue RESET reset_padValue STORED false)
    BR_PROPERTY(int, padSize, 0)
    BR_PROPERTY(int, padValue, 0)

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;

        foreach (const Mat &m, src) {
            Mat padded = padValue * Mat::ones(m.rows + 2*padSize, m.cols + 2*padSize, m.type());
            padded(Rect(padSize, padSize, padded.cols - padSize, padded.rows - padSize)) = m;
            dst += padded;
        }
    }
};

BR_REGISTER(Transform, PadTransform)

} // namespace br

#include "imgproc/pad.moc"
