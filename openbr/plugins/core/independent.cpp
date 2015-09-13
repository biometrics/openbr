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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Clones the Transform so that it can be applied independently.
 *
 * Independent Transforms expect single-matrix Templates.
 *
 * \author Josh Klontz \cite jklontz
 */
class IndependentTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform STORED false)
    BR_PROPERTY(br::Transform*, transform, NULL)

    QList<Transform*> transforms;

    QString description(bool expanded) const
    {
        return transform->description(expanded);
    }

    // can't use general setPropertyRecursive because of transforms oddness
    bool setPropertyRecursive(const QString &name, QVariant value)
    {
        if (br::Object::setExistingProperty(name, value))
            return true;

        if (!transform->setPropertyRecursive(name, value))
            return false;

        for (int i=0;i < transforms.size();i++)
            transforms[i]->setPropertyRecursive(name, value);

        return true;
    }

    Transform *simplify(bool &newTransform)
    {
        newTransform = false;
        bool newChild = false;
        Transform *temp = transform->simplify(newChild);
        if (temp == transform) {
            return this;
        }
        IndependentTransform* indep = new IndependentTransform();
        indep->transform = temp;

        IndependentTransform *test = dynamic_cast<IndependentTransform *> (temp);
        if (test) {
            // child was independent? this changes things...
            indep->transform = test->transform;
            for (int i=0; i < transforms.size(); i++) {
                bool newThing = false;
                IndependentTransform *probe = dynamic_cast<IndependentTransform *> (transforms[i]->simplify(newThing));
                indep->transforms.append(probe->transform);
                if (newThing)
                    probe->setParent(indep);
            }
            indep->file = indep->transform->file;
            indep->trainable = indep->transform->trainable;
            indep->setObjectName(indep->transform->objectName());

            return indep;
        }

        if (newChild)
            indep->transform->setParent(indep);

        for (int i=0; i < transforms.size();i++) {
            bool subTform = false;
            indep->transforms.append(transforms[i]->simplify(subTform));
            if (subTform)
                indep->transforms[i]->setParent(indep);
        }

        indep->file = indep->transform->file;
        indep->trainable = indep->transform->trainable;
        indep->setObjectName(indep->transform->objectName());

        return indep;
    }

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

    bool timeVarying() const { return transform->timeVarying(); }

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
                qWarning("Independent::train (%s) template %s of size %d differs from expected size %d.", qPrintable(objectName()), qPrintable(t.file.name), t.size(), templatesList.size());
            while (templatesList.size() < t.size())
                templatesList.append(TemplateList());
            for (int i=0; i<t.size(); i++)
                templatesList[i].append(Template(t.file, t[i]));
        }

        while (transforms.size() < templatesList.size())
            transforms.append(transform->clone());

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

    void projectUpdate(const Template &src, Template &dst)
    {
        dst.file = src.file;
        QList<Mat> mats;
        for (int i=0; i<src.size(); i++) {
            transforms[i%transforms.size()]->projectUpdate(Template(src.file, src[i]), dst);
            mats.append(dst);
            dst.clear();
        }
        dst.append(mats);
    }

    void finalize(TemplateList &out)
    {
        if (transforms.empty())
            return;

        transforms[0]->finalize(out);
        for (int i=1; i < transforms.size(); i++) {
            TemplateList temp;
            transforms[i]->finalize(temp);

            for (int j=0; j < out.size(); j++)
                out[j].append(temp[j]);
        }
    }

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst.reserve(src.size());
        foreach (const Template &t, src) {
            dst.append(Template());
            projectUpdate(t, dst.last());
        }
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

    QByteArray likely(const QByteArray &indentation) const
    {
        if (transforms.size() != 1)
            return "src"; // TODO: implement
        return transforms.first()->likely(indentation);
    }
};

BR_REGISTER(Transform, IndependentTransform)

} // namespace br

#include "core/independent.moc"
