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

#include <openbr/plugins/openbr_internal.h>

namespace br
{

static TemplateList Downsample(const TemplateList &templates, int classes, int instances, float fraction, const QString &inputVariable, const QStringList &gallery, const QStringList &subjects)
{
    // Return early when no downsampling is required
    if ((classes == std::numeric_limits<int>::max()) &&
            (instances == std::numeric_limits<int>::max()) &&
            (fraction >= 1) &&
            (gallery.isEmpty()) &&
            (subjects.isEmpty()))
        return templates;

    const bool atLeast = instances < 0;
    instances = abs(instances);

    QList<QString> allLabels = File::get<QString>(templates, inputVariable);

    QList<QString> uniqueLabels = allLabels.toSet().toList();
    qSort(uniqueLabels);

    QMap<QString,int> counts = templates.countValues<QString>(inputVariable, instances != std::numeric_limits<int>::max());

    if ((instances != std::numeric_limits<int>::max()) && (classes != std::numeric_limits<int>::max()))
        foreach (const QString &label, counts.keys())
            if (counts[label] < instances)
                counts.remove(label);

    uniqueLabels = counts.keys();
    if ((classes != std::numeric_limits<int>::max()) && (uniqueLabels.size() < classes))
        qWarning("Downsample requested %d classes but only %d are available.", classes, uniqueLabels.size());

    QList<QString> selectedLabels = uniqueLabels;
    if (classes < uniqueLabels.size()) {
        std::random_shuffle(selectedLabels.begin(), selectedLabels.end());
        selectedLabels = selectedLabels.mid(0, classes);
    }

    TemplateList downsample;
    for (int i=0; i<selectedLabels.size(); i++) {
        const QString selectedLabel = selectedLabels[i];
        QList<int> indices;
        for (int j=0; j<allLabels.size(); j++)
            if ((allLabels[j] == selectedLabel) && (!templates.value(j).file.get<bool>("FTE", false)) && (!templates.value(j).file.get<bool>("PossibleFTE", false)))
                indices.append(j);

        std::random_shuffle(indices.begin(), indices.end());
        const int max = atLeast ? indices.size() : std::min(indices.size(), instances);
        for (int j=0; j<max; j++)
            downsample.append(templates.value(indices[j]));
    }

    if (fraction < 1) {
        std::random_shuffle(downsample.begin(), downsample.end());
        downsample = downsample.mid(0, downsample.size()*fraction);
    }

    if (!gallery.isEmpty())
        for (int i=downsample.size()-1; i>=0; i--)
            if (!gallery.contains(downsample[i].file.get<QString>("Gallery")))
                downsample.removeAt(i);

    if (!subjects.isEmpty())
        for (int i=downsample.size()-1; i>=0; i--)
            if (subjects.contains(downsample[i].file.get<QString>(inputVariable)))
                downsample.removeAt(i);

    return downsample;
}

/*!
 * \ingroup transforms
 * \brief DOCUMENT ME JOSH
 * \author Josh Klontz \cite jklontz
 */
class DownsampleTrainingTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform STORED true)
    Q_PROPERTY(int classes READ get_classes WRITE set_classes RESET reset_classes STORED false)
    Q_PROPERTY(int instances READ get_instances WRITE set_instances RESET reset_instances STORED false)
    Q_PROPERTY(float fraction READ get_fraction WRITE set_fraction RESET reset_fraction STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(QStringList gallery READ get_gallery WRITE set_gallery RESET reset_gallery STORED false)
    Q_PROPERTY(QStringList subjects READ get_subjects WRITE set_subjects RESET reset_subjects STORED false)
    BR_PROPERTY(br::Transform*, transform, NULL)
    BR_PROPERTY(int, classes, std::numeric_limits<int>::max())
    BR_PROPERTY(int, instances, std::numeric_limits<int>::max())
    BR_PROPERTY(float, fraction, 1)
    BR_PROPERTY(QString, inputVariable, "Label")
    BR_PROPERTY(QStringList, gallery, QStringList())
    BR_PROPERTY(QStringList, subjects, QStringList())


    Transform *simplify(bool &newTForm)
    {
        Transform *res = transform->simplify(newTForm);
        return res;
    }

    void project(const Template &src, Template &dst) const
    {
       transform->project(src,dst);
    }


    void train(const TemplateList &data)
    {
        if (!transform || !transform->trainable)
            return;

        TemplateList downsampled = Downsample(data, classes, instances, fraction, inputVariable, gallery, subjects);

        transform->train(downsampled);
    }
};

BR_REGISTER(Transform, DownsampleTrainingTransform)

} // namespace br

#include "core/downsampletraining.moc"
