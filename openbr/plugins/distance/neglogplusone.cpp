#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Returns -log(distance(a,b)+1)
 * \author Josh Klontz \cite jklontz
 */
class NegativeLogPlusOneDistance : public UntrainableDistance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    BR_PROPERTY(br::Distance*, distance, NULL)

    void train(const TemplateList &src)
    {
        distance->train(src);
    }

    float compare(const Template &a, const Template &b) const
    {
        return -log(distance->compare(a,b)+1);
    }

    void store(QDataStream &stream) const
    {
        distance->store(stream);
    }

    void load(QDataStream &stream)
    {
        distance->load(stream);
    }
};

BR_REGISTER(Distance, NegativeLogPlusOneDistance)

} // namespace br

#include "neglogplusone.moc"
