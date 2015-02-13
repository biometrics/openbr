#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Online distance metric to attenuate match scores across multiple frames
 * \author Brendan klare \cite bklare
 */
class OnlineDistance : public UntrainableDistance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(float alpha READ get_alpha WRITE set_alpha RESET reset_alpha STORED false)
    BR_PROPERTY(br::Distance*, distance, NULL)
    BR_PROPERTY(float, alpha, 0.1f)

    mutable QHash<QString,float> scoreHash;
    mutable QMutex mutex;

    float compare(const Template &target, const Template &query) const
    {
        float currentScore = distance->compare(target, query);

        QMutexLocker mutexLocker(&mutex);
        return scoreHash[target.file.name] = (1.0- alpha) * scoreHash[target.file.name] + alpha * currentScore;
    }
};

BR_REGISTER(Distance, OnlineDistance)

} // namespace br

#include "online.moc"
