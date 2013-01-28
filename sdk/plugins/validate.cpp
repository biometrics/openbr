#include <QtConcurrentRun>
#include <openbr_plugin.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Cross validate a trainable transform.
 * \author Josh Klontz \cite jklontz
 */
class CrossValidateTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description STORED false)
    Q_PROPERTY(QList<br::Transform*> transforms READ get_transforms WRITE set_transforms RESET reset_transforms)
    BR_PROPERTY(QString, description, "Identity")
    BR_PROPERTY(QList<br::Transform*>, transforms, QList<br::Transform*>() << make(description))

    void train(const TemplateList &data)
    {
        if (!transforms.first()->trainable)
            return;

        int numPartitions = 0;
        QList<int> partitions; partitions.reserve(data.size());
        foreach (const File &file, data.files()) {
            partitions.append(file.getInt("Cross_Validation_Partition", 0));
            numPartitions = std::max(numPartitions, partitions.last());
        }

        if (numPartitions < 2) {
            transforms.first()->train(data);
            return;
        }

        while (transforms.size() < numPartitions)
            transforms.append(make(description));

        QList< QFuture<void> > futures;
        for (int i=0; i<numPartitions; i++) {
            qDebug() << "!!" << transforms[i]->description();
            TemplateList partitionedData = data;
            for (int j=partitionedData.size()-1; j>=0; j--)
                if (partitions[j] == i)
                    partitionedData.removeAt(j);
            if (Globals->parallelism) futures.append(QtConcurrent::run(transforms[i], &Transform::train, partitionedData));
            else                      transforms[i]->train(partitionedData);
        }
    }

    void project(const Template &src, Template &dst) const
    {
        transforms[src.file.getInt("Cross_Validation_Partition", 0)]->project(src, dst);
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
    Q_PROPERTY(br::Distance *distance READ get_distance WRITE set_distance RESET reset_distance)
    BR_PROPERTY(br::Distance*, distance, make("Dist(L2)"))

    void train(const TemplateList &src)
    {
        distance->train(src);
    }

    float compare(const Template &a, const Template &b) const
    {
        const int partitionA = a.file.getInt("Cross_Validation_Partition", 0);
        const int partitionB = b.file.getInt("Cross_Validation_Partition", 0);
        if (partitionA != partitionB) return -std::numeric_limits<float>::max();
        return distance->compare(a, b);
    }
};

BR_REGISTER(Distance, CrossValidateDistance)

} // namespace br

#include "validate.moc"
