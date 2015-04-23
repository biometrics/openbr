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

/*!
 * \ingroup transforms
 * \brief A globally shared Transform.
 * \author Josh Klontz \cite jklontz
 */
class SingletonTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description STORED false)
    BR_PROPERTY(QString, description, "Identity")

    static QMutex mutex;
    static QHash<QString,Transform*> transforms;
    static QHash<QString,int> trainingReferenceCounts;
    static QHash<QString,TemplateList> trainingData;

    Transform *transform;

    void init()
    {
        QMutexLocker locker(&mutex);
        if (!transforms.contains(description)) {
            transforms.insert(description, make(description));
            trainingReferenceCounts.insert(description, 0);
        }

        transform = transforms[description];
        trainingReferenceCounts[description]++;
    }

    void train(const TemplateList &data)
    {
        QMutexLocker locker(&mutex);
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

QMutex SingletonTransform::mutex;
QHash<QString,Transform*> SingletonTransform::transforms;
QHash<QString,int> SingletonTransform::trainingReferenceCounts;
QHash<QString,TemplateList> SingletonTransform::trainingData;

BR_REGISTER(Transform, SingletonTransform)

} // namespace br

#include "core/singleton.moc"
