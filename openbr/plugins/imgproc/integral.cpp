#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Computes integral image.
 * \author Josh Klontz \cite jklontz
 */
class IntegralTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        cv::integral(src, dst);
    }
};

BR_REGISTER(Transform, IntegralTransform)

} // namespace br

#include "integral.moc"
