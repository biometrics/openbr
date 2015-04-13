/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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
 * \note  Two flags can be put in File metadata that are related to cross-validation and are used to
 *        extend a testing gallery:
 *        (i) allPartitions - This flag is intended to be used when comparing the
 *                            performance of an untrainable algorithm (e.g. a COTS
 *                            algorithm) against a trainable algorithm that was trained
 *                            using cross-validation. All templates with the allPartitions
 *                            flag will be compared against for every partition.  As
 *                            untrainable algorithms will have no use for the
 *                            CrossValidateTransform, this flag is only meaningful at comparison
 *                            time (but care has been taken so that one can train and enroll
 *                            without issue if these Files are present in the used Gallery).
 *        (ii) duplicatePartitions - This flag is similar to allPartitions in that it causes
 *                            the same template to be used during comparison for every partition.
 *                            The difference is that duplicatePartitions will duplicate each
 *                            marked template and project it into the model space constituded
 *                            by the child transforms of CrossValidateTransform.  Again, care
 *                            has been take such that one can train with these templates in the
 *                            used Gallery successfully (they will simply be omitted).
 */
class CrossValidateTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description STORED false)
    BR_PROPERTY(QString, description, "Identity")

    // numPartitions copies of transform specified by description.
    QList<br::Transform*> transforms;

    // Treating this transform as a leaf (in terms of updated training scheme), the child transform
    // of this transform will lose any structure present in the training QList<TemplateList>, which
    // is generally incorrect behavior.
    void train(const TemplateList &data)
    {
        QList<int> partitions = data.files().crossValidationPartitions();
        const int numPartitions = Common::Max(partitions)+1;

        while (transforms.size() < numPartitions)
            transforms.append(make(description));

        if (numPartitions < 2) {
            transforms.first()->train(data);
            return;
        }

        QFutureSynchronizer<void> futures;
        for (int i=0; i<numPartitions; i++) {
            TemplateList partitionedData = data;
            for (int j=partitionedData.size()-1; j>=0; j--)
                if (partitions[j] == i) {
                    // Remove data, it's designated for testing
                    partitionedData.removeAt(j);
                }
            // Train on the remaining templates
            futures.addFuture(QtConcurrent::run(_train, transforms[i], partitionedData));
        }
        futures.waitForFinished();
    }

    void project(const Template &src, Template &dst) const
    {
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
