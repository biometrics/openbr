#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/distance_sse.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup distances
 * \brief Fast 4-bit L1 distance
 * \author Josh Klontz \cite jklontz
 */
class HalfByteL1Distance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const Mat &a, const Mat &b) const
    {
        return packed_l1(a.data, b.data, a.total());
    }
};

BR_REGISTER(Distance, HalfByteL1Distance)


} // namespace br

#include "halfbyteL1.moc"
