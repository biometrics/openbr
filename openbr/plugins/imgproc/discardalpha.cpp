#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Drop the alpha channel (if exists).
 * \author Austin Blanton \cite imaus10
 */
class DiscardAlphaTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() > 4 || src.m().channels() == 2) {
            dst.file.fte = true;
            return;
        }

        dst = src;
        if (src.m().channels() == 4) {
            std::vector<Mat> mv;
            split(src, mv);
            mv.pop_back();
            merge(mv, dst);
        }
    }
};

BR_REGISTER(Transform, DiscardAlphaTransform)

} // namespace br

#include "imgproc/discardalpha.moc"
