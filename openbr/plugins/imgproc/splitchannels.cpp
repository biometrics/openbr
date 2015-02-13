#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Split a multi-channel matrix into several single-channel matrices.
 * \author Josh Klontz \cite jklontz
 */
class SplitChannelsTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        std::vector<Mat> mv;
        split(src, mv);
        foreach (const Mat &m, mv)
            dst += m;
    }
};

BR_REGISTER(Transform, SplitChannelsTransform)

} // namespace br

#include "splitchannels.moc"
