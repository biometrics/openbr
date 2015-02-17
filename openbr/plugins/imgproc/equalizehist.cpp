#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Histogram equalization
 * \author Josh Klontz \cite jklontz
 */
class EqualizeHistTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        cv::equalizeHist(src, dst);
    }
};

BR_REGISTER(Transform, EqualizeHistTransform)

} // namespace br

#include "imgproc/equalizehist.moc"
