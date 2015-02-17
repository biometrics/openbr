#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup distances
 * \brief Returns \c true if the templates are identical, \c false otherwise.
 * \author Josh Klontz \cite jklontz
 */
class IdenticalDistance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const Mat &a, const Mat &b) const
    {
        const size_t size = a.total() * a.elemSize();
        if (size != b.total() * b.elemSize()) return 0;
        for (size_t i=0; i<size; i++)
            if (a.data[i] != b.data[i]) return 0;
        return 1;
    }
};

BR_REGISTER(Distance, IdenticalDistance)

} // namespace br

#include "distance/identical.moc"
