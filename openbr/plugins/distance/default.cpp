#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief DistDistance wrapper.
 * \author Josh Klontz \cite jklontz
 */
class DefaultDistance : public UntrainableDistance
{
    Q_OBJECT
    Distance *distance;

    void init()
    {
        distance = Distance::make("Dist("+file.suffix()+")");
    }

    float compare(const cv::Mat &a, const cv::Mat &b) const
    {
        return distance->compare(a, b);
    }
};

BR_REGISTER(Distance, DefaultDistance)

} // namespace br

#include "default.moc"
