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
 * \brief Store the last matrix of the input Template as a metadata key with input property name.
 * \author Charles Otto \cite caotto
 */
class SaveMatTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(QString propName READ get_propName WRITE set_propName RESET reset_propName STORED false)
    BR_PROPERTY(QString, propName, "")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.file.set(propName, QVariant::fromValue(dst.m()));
    }

    void docs(int indent) const
    {
        print_doc_header("Save the template matrix into the metadata under the specified key", indent);
    }
};

BR_REGISTER(Transform, SaveMatTransform)

/*!
 * \ingroup transforms
 * \brief Preserve the contents of the input template, updating just the specified metadata keys.
 * \author Josh Klontz \cite jklontz
 */
class JustTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform)
    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    BR_PROPERTY(br::Transform*, transform, NULL)
    BR_PROPERTY(QStringList, keys, QStringList())

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        Template tmp;
        transform->project(src, tmp);
        foreach (const QString &key, keys)
            if (key == "_Points") {
                dst.file.setPoints(tmp.file.points());
            } else if (tmp.file.contains(key)) {
                dst.file.set(key, tmp.file.value(key));
            }
    }

    void docs(int indent) const
    {
        print_doc_header("Preserve the input template and only updated the specified keys", indent);
        if (!transform) return; // If no inner transform, nothing to do.
        // We need to create a new instance of the transform for any independent transforms
        // because they are wrapped by MetaTransform until project() is called.
        QString description = "." + transform->description(false); // Needs to start with a .
        Factory<Transform>::make_docs(description)->docs(indent + 4);
    }
};

BR_REGISTER(Transform, JustTransform)

} // namespace br

#include "metadata/savemat.moc"
