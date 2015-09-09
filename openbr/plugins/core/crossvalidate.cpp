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
 * \brief Cross validate a trainable Transform.
 *
 * Two flags can be put in File metadata that are related to cross-validation and are used to
 * extend a testing gallery:
 *
 *     flag | description
 *     --- | ---
 *     allPartitions | This flag is intended to be used when comparing the performance of an untrainable algorithm (e.g. a COTS algorithm) against a trainable algorithm that was trained using cross-validation. All templates with the allPartitions flag will be compared against for every partition. As untrainable algorithms will have no use for the CrossValidateTransform, this flag is only meaningful at comparison time (but care has been taken so that one can train and enroll without issue if these Files are present in the used Gallery).
 *     duplicatePartitions | This flag is similar to allPartitions in that it causes the same template to be used during comparison for every partition. The difference is that duplicatePartitions will duplicate each marked template and project it into the model space constituded by the child transforms of CrossValidateTransform. Again, care has been take such that one can train with these templates in the used Gallery successfully (they will simply be omitted).
 *
 * To use an extended Gallery, add an allPartitions="true" flag to the gallery sigset for those images that should be compared
 * against for all testing partitions.
 *
 * \author Josh Klontz \cite jklontz
 * \author Scott Klum \cite sklum
 */
class CrossValidateTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(unsigned int randomSeed READ get_randomSeed WRITE set_randomSeed RESET reset_randomSeed STORED false)
    BR_PROPERTY(QString, description, "Identity")
    BR_PROPERTY(QString, inputVariable, "Label")
    BR_PROPERTY(unsigned int, randomSeed, 0)

    // numPartitions copies of transform specified by description.
    QList<br::Transform*> transforms;

    // Treating this transform as a leaf (in terms of updated training scheme), the child transform
    // of this transform will lose any structure present in the training QList<TemplateList>, which
    // is generally incorrect behavior.
    void train(const TemplateList &data)
    {
        TemplateList partitionedData = data.partition(inputVariable, randomSeed, true);
        QList<int> partitions = partitionedData.files().crossValidationPartitions();

        const int crossValidate = Globals->crossValidate;
        // Only train once based on the 0th partition if crossValidate is negative.
        const int numPartitions = (crossValidate < 0) ? 1 : Common::Max(partitions)+1;
        while (transforms.size() < numPartitions)
            transforms.append(make(description));

        if (std::abs(crossValidate) < 2) {
            transforms.first()->train(data);
            return;
        }

        QFutureSynchronizer<void> futures;
        for (int i=0; i<numPartitions; i++) {
            TemplateList partition = partitionedData;
            for (int j=partition.size()-1; j>=0; j--) {
                if (partitions[j] == i)
                    // Remove data, it's designated for testing
                    partition.removeAt(j);
            }
            if (Globals->verbose)
                qDebug() << QString("Training partition %1 on %2 templates.").arg(QString::number(i),QString::number(partition.size()));

            // Train on the remaining templates
            futures.addFuture(QtConcurrent::run(_train, transforms[i], partition));
        }
        futures.waitForFinished();
    }

    void project(const Template &src, Template &dst) const
    {
        Q_UNUSED(src);
        Q_UNUSED(dst);

        qFatal("CrossValidateTransform::project(const Template &src, Template &dst) should not be called.");
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        TemplateList partitioned = src.partition(inputVariable, randomSeed, true);
        const int crossValidate = Globals->crossValidate;

        if (crossValidate < 0) {
            transforms[0]->project(partitioned, dst);
            return;
        }
        for (int i=0; i<partitioned.size(); i++) {
            int partition = partitioned[i].file.get<int>("Partition", 0);
            transforms[partition]->project(partitioned, dst);
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

} // namespace br

#include "core/crossvalidate.moc"
