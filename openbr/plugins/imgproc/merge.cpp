#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV merge
 * \author Josh Klontz \cite jklontz
 */
class MergeTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        std::vector<Mat> mv;
        foreach (const Mat &m, src)
            mv.push_back(m);
        merge(mv, dst);
    }
};

BR_REGISTER(Transform, MergeTransform)

} // namespace br

#include "imgproc/merge.moc"
