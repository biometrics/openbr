#include <QFutureSynchronizer>
#include <QtConcurrentRun>
#include "openbr_internal.h"
#include "openbr/core/common.h"
#include <openbr/core/qtutils.h>

namespace br
{

static void _train(Transform * transform, TemplateList data) // think data has to be a copy -cao
{
    transform->train(data);
}

/*!
 * \ingroup transforms
 * \brief Cross validate a trainable transform.
 * \author Josh Klontz \cite jklontz
 * \author Scott Klum \cite sklum
 * \note To use an extended gallery, add an allPartitions="true" flag to the gallery sigset for those images that should be compared
 *       against for all testing partitions.
 */
class CrossValidateTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description STORED false)
    Q_PROPERTY(bool leaveOneImageOut READ get_leaveOneImageOut WRITE set_leaveOneImageOut RESET reset_leaveOneImageOut STORED false)
    BR_PROPERTY(QString, description, "Identity")
    BR_PROPERTY(bool, leaveOneImageOut, false)

    // numPartitions copies of transform specified by description.
    QList<br::Transform*> transforms;

    // Treating this transform as a leaf (in terms of update training scheme), the child transform
    // of this transform will lose any structure present in the training QList<TemplateList>, which
    // is generally incorrect behavior.
    void train(const TemplateList &data)
    {
        int numPartitions = 0;
        QList<int> partitions; partitions.reserve(data.size());
        foreach (const File &file, data.files()) {
            partitions.append(file.get<int>("Partition", 0));
            numPartitions = std::max(numPartitions, partitions.last()+1);
        }

        while (transforms.size() < numPartitions)
            transforms.append(make(description));

        if (numPartitions < 2) {
            transforms.first()->train(data);
            return;
        }

        QFutureSynchronizer<void> futures;
        for (int i=0; i<numPartitions; i++) {
            QList<int> partitionsBuffer = partitions;
            TemplateList partitionedData = data;
            int j = partitionedData.size()-1;
            while (j>=0) {
                // Remove all templates belonging to partition i
                // if leaveOneImageOut is true,
                // and i is greater than the number of images for a particular subject
                // even if the partitions are different
                if (leaveOneImageOut) {
                    const QString label = partitionedData.at(j).file.get<QString>("Label");
                    QList<int> subjectIndices = partitionedData.find("Label",label);
                    QList<int> removed;
                    // Remove test only data
                    for (int k=subjectIndices.size()-1; k>=0; k--)
                        if (partitionedData[subjectIndices[k]].file.getBool("testOnly")) {
                            removed.append(subjectIndices[k]);
                            subjectIndices.removeAt(k);
                        }
                    // Remove template that was repeated to make the testOnly template
                    if (subjectIndices.size() > 1 && subjectIndices.size() <= i) {
                        removed.append(subjectIndices[i%subjectIndices.size()]);
                    } else if (partitionsBuffer[j] == i) {
                        removed.append(j);
                    }

                    if (!removed.empty()) {
                        typedef QPair<int,int> Pair;
                        foreach (Pair pair, Common::Sort(removed,true)) {
                            partitionedData.removeAt(pair.first); partitionsBuffer.removeAt(pair.first); j--;
                        }
                    } else {
                        j--;
                    }
                } else if (partitions[j] == i) {
                    // Remove data, it's designated for testing
                    partitionedData.removeAt(j);
                    j--;
                } else j--;
            }
            // Train on the remaining templates
            futures.addFuture(QtConcurrent::run(_train, transforms[i], partitionedData));
        }
        futures.waitForFinished();
    }

    void project(const Template &src, Template &dst) const
    {
        if (src.file.getBool("Train", false)) dst = src;
        else {
            // If we want to duplicate templates but use the same training data
            // for all partitions (i.e. transforms.size() == 1), we need to
            // restrict the partition
            int partition = src.file.get<int>("Partition", 0);
            partition = (partition >= transforms.size()) ? 0 : partition;
            transforms[partition]->project(src, dst);
        }
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
        static const QString key("Partition"); // More efficient to preallocate this
        const int partitionA = a.file.get<int>(key, 0);
        const int partitionB = b.file.get<int>(key, 0);
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
            bool keep = false;
            const QString metadata = a.file.get<QString>(key, "");
            if (Globals->filters[key].isEmpty()) continue;
            if (metadata.isEmpty()) return -std::numeric_limits<float>::max();
            foreach (const QString &value, Globals->filters[key]) {
                if (metadata == value) {
                    keep = true;
                    break;
                }
            }
            if (!keep) return -std::numeric_limits<float>::max();
        }
        return 0;
    }
};

BR_REGISTER(Distance, FilterDistance)

/*!
 * \ingroup distances
 * \brief Checks target metadata against query metadata.
 * \author Scott Klum \cite sklum
 */
class MetadataDistance : public Distance
{
    Q_OBJECT

    Q_PROPERTY(QStringList filters READ get_filters WRITE set_filters RESET reset_filters STORED false)
    BR_PROPERTY(QStringList, filters, QStringList())

    float compare(const Template &a, const Template &b) const
    {
        foreach (const QString &key, filters) {
            QString aValue = a.file.get<QString>(key, QString());
            QString bValue = b.file.get<QString>(key, QString());

            // The query value may be a range. Let's check.
            if (bValue.isEmpty()) bValue = QtUtils::toString(b.file.get<QPointF>(key, QPointF()));

            if (aValue.isEmpty() || bValue.isEmpty()) continue;

            bool keep = false;
            bool ok;

            QPointF range = QtUtils::toPoint(bValue,&ok);

            if (ok) /* Range */ {
                int value = range.x();
                int upperBound = range.y();

                while (value <= upperBound) {
                    if (aValue == QString::number(value)) {
                        keep = true;
                        break;
                    }
                    value++;
                }
            }
            else if (aValue == bValue) keep = true;

            if (!keep) return -std::numeric_limits<float>::max();
        }
        return 0;
    }
};


BR_REGISTER(Distance, MetadataDistance)

/*!
 * \ingroup distances
 * \brief Sets distance to -FLOAT_MAX if a target template has/doesn't have a key.
 * \author Scott Klum \cite sklum
 */
class RejectDistance : public Distance
{
    Q_OBJECT

    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    BR_PROPERTY(QStringList, keys, QStringList())
    Q_PROPERTY(bool rejectIfContains READ get_rejectIfContains WRITE set_rejectIfContains RESET reset_rejectIfContains STORED false)
    BR_PROPERTY(bool, rejectIfContains, false)

    float compare(const Template &a, const Template &b) const
    {
        (void) b;
        bool keep = true;

        foreach (const QString &key, keys) {
            if ((rejectIfContains && a.file.contains(key)) ||
                (!rejectIfContains && !a.file.contains(key)))
                keep = false;

            if (!keep) return -std::numeric_limits<float>::max();
        }

        return 0;
    }
};


BR_REGISTER(Distance, RejectDistance)

} // namespace br

#include "validate.moc"
