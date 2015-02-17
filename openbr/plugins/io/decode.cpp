#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Decodes images
 * \author Josh Klontz \cite jklontz
 */
class DecodeTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.append(cv::imdecode(src.m(), cv::IMREAD_UNCHANGED));
    }
};

BR_REGISTER(Transform, DecodeTransform)

} // namespace br

#include "io/decode.moc"
