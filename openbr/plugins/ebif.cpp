#include "openbr_internal.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Face Recognition Using Early Biologically Inspired Features
 * Min Li (IBM China Research Lab, China), Nalini Ratha (IBM Watson Research Center,
 * USA), Weihong Qian (IBM China Research Lab, China), Shenghua Bao (IBM China
 * Research Lab, China), Zhong Su (IBM China Research Lab, China)
 * \author Josh Klontz \cite jklontz
 */

class EBIFTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        (void) src;
        (void) dst;
    }
};

BR_REGISTER(Transform, EBIFTransform)

} // namespace br

#include "ebif.moc"
