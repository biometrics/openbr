#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Convert to uchar format
 * \author Josh Klontz \cite jklontz
 */
class CvtUCharTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        OpenCVUtils::cvtUChar(src, dst);
    }
};

BR_REGISTER(Transform, CvtUCharTransform)

} // namespace br

#include "cvtuchar.moc"
