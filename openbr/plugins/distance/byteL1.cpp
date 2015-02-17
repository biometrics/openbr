#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/distance_sse.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Fast 8-bit L1 distance
 * \author Josh Klontz \cite jklontz
 */
class ByteL1Distance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const unsigned char *a, const unsigned char *b, size_t size) const
    {
        return l1(a, b, size);
    }
};

BR_REGISTER(Distance, ByteL1Distance)

} // namespace br

#include "distance/byteL1.moc"
