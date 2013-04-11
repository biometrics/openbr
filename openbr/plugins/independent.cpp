#include <QFutureSynchronizer>
#include <QtConcurrentRun>

#include "openbr_internal.h"
#include "openbr/core/common.h"

using namespace cv;

namespace br
{

static TemplateList Downsample(const TemplateList &templates, const Transform *transform)
{
    // Return early when no downsampling is required
    if ((transform->classes == std::numeric_limits<int>::max()) &&
            (transform->instances == std::numeric_limits<int>::max()) &&
            (transform->fraction >= 1))
        return templates;

    const bool atLeast = transform->instances < 0;
    const int instances = abs(transform->instances);

    QList<int> allLabels = templates.labels<int>();
    QList<int> uniqueLabels = allLabels.toSet().toList();
    qSort(uniqueLabels);

    QMap<int,int> counts = templates.labelCounts(instances != std::numeric_limits<int>::max());
    if ((instances != std::numeric_limits<int>::max()) && (transform->classes != std::numeric_limits<int>::max()))
        foreach (int label, counts.keys())
            if (counts[label] < instances)
                counts.remove(label);
    uniqueLabels = counts.keys();
    if ((transform->classes != std::numeric_limits<int>::max()) && (uniqueLabels.size() < transform->classes))
        qWarning("Downsample requested %d classes but only %d are available.", transform->classes, uniqueLabels.size());

    Common::seedRNG();
    QList<int> selectedLabels = uniqueLabels;
    if (transform->classes < uniqueLabels.size()) {
        std::random_shuffle(selectedLabels.begin(), selectedLabels.end());
        selectedLabels = selectedLabels.mid(0, transform->classes);
    }

    TemplateList downsample;
    for (int i=0; i<selectedLabels.size(); i++) {
        const int selectedLabel = selectedLabels[i];
        QList<int> indices;
        for (int j=0; j<allLabels.size(); j++)
            if ((allLabels[j] == selectedLabel) && (!templates.value(j).file.get<bool>("FTE", false)))
                indices.append(j);

        std::random_shuffle(indices.begin(), indices.end());
        const int max = atLeast ? indices.size() : std::min(indices.size(), instances);
        for (int j=0; j<max; j++)
            downsample.append(templates.value(indices[j]));
    }

    if (transform->fraction < 1) {
        std::random_shuffle(downsample.begin(), downsample.end());
        downsample = downsample.mid(0, downsample.size()*transform->fraction);
    }

    return downsample;
}

/*!
 * \ingroup transforms
 * \brief Clones the transform so that it can be applied independently.
 * \author Josh Klontz \cite jklontz
 * \em Independent transforms expect single-matrix templates.
 */
class IndependentTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform STORED false)
    BR_PROPERTY(br::Transform*, transform, NULL)

    QList<Transform*> transforms;

    void init()
    {
        transforms.clear();
        if (transform == NULL)
            return;

        transform->setParent(this);
        transforms.append(transform);
        file = transform->file;
        trainable = transform->trainable;
        setObjectName(transform->objectName());
    }

    Transform *clone() const
    {
        IndependentTransform *independentTransform = new IndependentTransform();
        independentTransform->transform = transform->clone();
        independentTransform->init();
        return independentTransform;
    }

    static void _train(Transform *transform, const TemplateList *data)
    {
        transform->train(*data);
    }

    void train(const TemplateList &data)
    {
        // Don't bother if the transform is untrainable
        if (!trainable) return;

        QList<TemplateList> templatesList;
        foreach (const Template &t, data) {
            if ((templatesList.size() != t.size()) && !templatesList.isEmpty())
                qWarning("Independent::train template %s of size %d differs from expected size %d.", qPrintable(t.file.name), t.size(), templatesList.size());
            while (templatesList.size() < t.size())
                templatesList.append(TemplateList());
            for (int i=0; i<t.size(); i++)
                templatesList[i].append(Template(t.file, t[i]));
        }

        while (transforms.size() < templatesList.size())
            transforms.append(transform->clone());

        for (int i=0; i<templatesList.size(); i++)
            templatesList[i] = Downsample(templatesList[i], transforms[i]);

        QFutureSynchronizer<void> futures;
        for (int i=0; i<templatesList.size(); i++)
            futures.addFuture(QtConcurrent::run(_train, transforms[i], &templatesList[i]));
        futures.waitForFinished();
    }

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        QList<Mat> mats;
        for (int i=0; i<src.size(); i++) {
            transforms[i%transforms.size()]->project(Template(src.file, src[i]), dst);
            mats.append(dst);
            dst.clear();
        }
        dst.append(mats);
    }

    void store(QDataStream &stream) const
    {
        const int size = transforms.size();
        stream << size;
        for (int i=0; i<size; i++)
            transforms[i]->store(stream);
    }

    void load(QDataStream &stream)
    {
        int size;
        stream >> size;
        while (transforms.size() < size)
            transforms.append(transform->clone());
        for (int i=0; i<size; i++)
            transforms[i]->load(stream);
    }
};

BR_REGISTER(Transform, IndependentTransform)

/*!
 * \ingroup transforms
 * \brief A globally shared transform.
 * \author Josh Klontz \cite jklontz
 */
class SingletonTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description STORED false)
    BR_PROPERTY(QString, description, "Identity")

    static QHash<QString,Transform*> transforms;
    static QMutex trainingMutex;
    static QHash<QString,int> trainingReferenceCounts;
    static QHash<QString,TemplateList> trainingData;

    Transform *transform;

    void init()
    {
        if (!transforms.contains(description)) {
            transforms.insert(description, make(description));
            trainingReferenceCounts.insert(description, 0);
        }

        transform = transforms[description];
        trainingReferenceCounts[description]++;
    }

    void train(const TemplateList &data)
    {
        QMutexLocker locker(&trainingMutex);
        trainingData[description].append(data);
        trainingReferenceCounts[description]--;
        if (trainingReferenceCounts[description] > 0) return;
        transform->train(trainingData[description]);
        trainingData[description].clear();
    }

    void project(const Template &src, Template &dst) const
    {
        transform->project(src, dst);
    }

    void store(QDataStream &stream) const
    {
        if (transform->parent() == this)
            transform->store(stream);
    }

    void load(QDataStream &stream)
    {
        if (transform->parent() == this)
            transform->load(stream);
    }
};

QHash<QString,Transform*> SingletonTransform::transforms;
QMutex SingletonTransform::trainingMutex;
QHash<QString,int> SingletonTransform::trainingReferenceCounts;
QHash<QString,TemplateList> SingletonTransform::trainingData;

BR_REGISTER(Transform, SingletonTransform)

} // namespace br

#include "independent.moc"
