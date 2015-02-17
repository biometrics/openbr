#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Computes the absolute value of each element.
 * \author Josh Klontz \cite jklontz
 */
class AbsTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = cv::abs(src);
    }
};

BR_REGISTER(Transform, AbsTransform)

} // namespace br

#include "imgproc/abs.moc"
