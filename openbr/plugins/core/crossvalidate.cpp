#include <QtConcurrent>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>

namespace br
{

static void _train(Transform *transform, TemplateList data) // think data has to be a copy -cao
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
                    // Remove target only data
                    for (int k=subjectIndices.size()-1; k>=0; k--)
                        if (partitionedData[subjectIndices[k]].file.getBool("targetOnly")) {
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
        // Remember, the partition should never be -1
        // since it is assumed that the allPartitions
        // flag is only used during comparison
        // (i.e. only used when making a mask)

        // If we want to duplicate templates but use the same training data
        // for all partitions (i.e. transforms.size() == 1), we need to
        // restrict the partition

        int partition = src.file.get<int>("Partition", 0);
        partition = (partition >= transforms.size()) ? 0 : partition;
        transforms[partition]->project(src, dst);
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

} // namespace br

#include "core/crossvalidate.moc"
