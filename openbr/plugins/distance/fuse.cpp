#include <numeric>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Fuses similarity scores across multiple matrices of compared templates
 * \author Scott Klum \cite sklum
 * \note Operation: Mean, sum, min, max are supported.
 */
class FuseDistance : public Distance
{
    Q_OBJECT
    Q_ENUMS(Operation)
    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description STORED false)
    Q_PROPERTY(Operation operation READ get_operation WRITE set_operation RESET reset_operation STORED false)
    Q_PROPERTY(QList<float> weights READ get_weights WRITE set_weights RESET reset_weights STORED false)

    QList<br::Distance*> distances;

public:
    /*!< */
    enum Operation {Mean, Sum, Max, Min};

private:
    BR_PROPERTY(QString, description, "L2")
    BR_PROPERTY(Operation, operation, Mean)
    BR_PROPERTY(QList<float>, weights, QList<float>())

    void train(const TemplateList &src)
    {
        // Partition the templates by matrix
        QList<int> split;
        for (int i=0; i<src.at(0).size(); i++) split.append(1);

        QList<TemplateList> partitionedSrc = src.partition(split);

        while (distances.size() < partitionedSrc.size())
            distances.append(make(description));

        // Train on each of the partitions
        for (int i=0; i<distances.size(); i++)
            distances[i]->train(partitionedSrc[i]);
    }

    float compare(const Template &a, const Template &b) const
    {
        if (a.size() != b.size()) qFatal("Comparison size mismatch");

        QList<float> scores;
        for (int i=0; i<distances.size(); i++) {
            float weight;
            weights.isEmpty() ? weight = 1. : weight = weights[i];
            scores.append(weight*distances[i]->compare(Template(a.file, a[i]),Template(b.file, b[i])));
        }

        switch (operation) {
          case Mean:
            return std::accumulate(scores.begin(),scores.end(),0.0)/(float)scores.size();
            break;
          case Sum:
            return std::accumulate(scores.begin(),scores.end(),0.0);
            break;
          case Min:
            return *std::min_element(scores.begin(),scores.end());
            break;
          case Max:
            return *std::max_element(scores.begin(),scores.end());
            break;
          default:
            qFatal("Invalid operation.");
        }
        return 0;
    }

    void store(QDataStream &stream) const
    {
        stream << distances.size();
        foreach (Distance *distance, distances)
            distance->store(stream);
    }

    void load(QDataStream &stream)
    {
        int numDistances;
        stream >> numDistances;
        while (distances.size() < numDistances)
            distances.append(make(description));
        foreach (Distance *distance, distances)
            distance->load(stream);
    }
};

BR_REGISTER(Distance, FuseDistance)

} // namespace br

#include "fuse.moc"
