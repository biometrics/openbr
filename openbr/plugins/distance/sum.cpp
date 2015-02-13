#include <QtConcurrent>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Sum match scores across multiple distances
 * \author Scott Klum \cite sklum
 */
class SumDistance : public UntrainableDistance
{
    Q_OBJECT
    Q_PROPERTY(QList<br::Distance*> distances READ get_distances WRITE set_distances RESET reset_distances)
    BR_PROPERTY(QList<br::Distance*>, distances, QList<br::Distance*>())

    void train(const TemplateList &data)
    {
        QFutureSynchronizer<void> futures;
        foreach (br::Distance *distance, distances)
            futures.addFuture(QtConcurrent::run(distance, &Distance::train, data));
        futures.waitForFinished();
    }

    float compare(const Template &target, const Template &query) const
    {
        float result = 0;

        foreach (br::Distance *distance, distances) {
            result += distance->compare(target, query);

            if (result == -std::numeric_limits<float>::max())
                return result;
        }

        return result;
    }
};

BR_REGISTER(Distance, SumDistance)

} // namespace br

#include "sum.moc"
