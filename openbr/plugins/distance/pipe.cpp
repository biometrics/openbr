#include <QtConcurrent>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Distances in series.
 * \author Josh Klontz \cite jklontz
 *
 * The templates are compared using each br::Distance in order.
 * If the result of the comparison with any given distance is -FLOAT_MAX then this result is returned early.
 * Otherwise the returned result is the value of comparing the templates using the last br::Distance.
 */
class PipeDistance : public Distance
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

    float compare(const Template &a, const Template &b) const
    {
        float result = -std::numeric_limits<float>::max();
        foreach (br::Distance *distance, distances) {
            result = distance->compare(a, b);
            if (result == -std::numeric_limits<float>::max())
                return result;
        }
        return result;
    }
};

BR_REGISTER(Distance, PipeDistance)

} // namespace br

#include "distance/pipe.moc"
