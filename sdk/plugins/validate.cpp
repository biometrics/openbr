#include <QtConcurrentRun>
#include <openbr_plugin.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Cross validate a trainable transform.
 * \author Josh Klontz \cite jklontz
 */
class CrossValidateTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description STORED false)
    BR_PROPERTY(QString, description, "Identity")

    QList<br::Transform*> transforms;

    void train(const TemplateList &data)
    {
        int numPartitions = 0;
        QList<int> partitions; partitions.reserve(data.size());
        foreach (const File &file, data.files()) {
            partitions.append(file.getInt("Cross_Validation_Partition", 0));
            numPartitions = std::max(numPartitions, partitions.last()+1);
        }

        while (transforms.size() < numPartitions)
            transforms.append(make(description));

        if (numPartitions < 2) {
            transforms.first()->train(data);
            return;
        }

        QList< QFuture<void> > futures;
        for (int i=0; i<numPartitions; i++) {
            TemplateList partitionedData = data;
            for (int j=partitionedData.size()-1; j>=0; j--)
                if (partitions[j] == i)
                    partitionedData.removeAt(j);
            if (Globals->parallelism) futures.append(QtConcurrent::run(transforms[i], &Transform::train, partitionedData));
            else                      transforms[i]->train(partitionedData);
        }
        Globals->trackFutures(futures);
    }

    void project(const Template &src, Template &dst) const
    {
        transforms[src.file.getInt("Cross_Validation_Partition", 0)]->project(src, dst);
    }

    void store(QDataStream &stream) const
    {
        stream << transforms.size();
        foreach (Transform *transform, transforms)
            transform->store(stream);
    }

    void load(QDataStream &stream)
    {
        int numTransforms;
        stream >> numTransforms;
        while (transforms.size() < numTransforms)
            transforms.append(make(description));
        foreach (Transform *transform, transforms)
            transform->load(stream);
    }
};

BR_REGISTER(Transform, CrossValidateTransform)

/*!
 * \ingroup distances
 * \brief Cross validate a distance metric.
 * \author Josh Klontz \cite jklontz
 */
class CrossValidateDistance : public Distance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        const int partitionA = a.file.getInt("Cross_Validation_Partition", 0);
        const int partitionB = b.file.getInt("Cross_Validation_Partition", 0);
        return (partitionA != partitionB) ? -std::numeric_limits<float>::max() : 0;
    }
};

BR_REGISTER(Distance, CrossValidateDistance)

/*!
 * \ingroup distances
 * \brief Checks target metadata against filters.
 * \author Josh Klontz \cite jklontz
 */
class FilterDistance : public Distance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        (void) b; // Query template isn't checked
        foreach (const QString &key, Globals->filters.keys()) {
            const QString metadata = a.file.getString(key, "");
            if (metadata.isEmpty()) continue;
            foreach (const QString &value, Globals->filters[key]) {
                if (metadata == value) continue;
                return -std::numeric_limits<float>::max();
            }
        }
        return 0;
    }
};

BR_REGISTER(Distance, FilterDistance)

} // namespace br

#include "validate.moc"
