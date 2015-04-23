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
 * \brief Removes duplicate Templates based on a unique metadata key
 * \author Austin Blanton \cite imaus10
 */
class FilterDupeMetadataTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    BR_PROPERTY(QString, key, "TemplateID")

    QSet<QString> excluded;

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        foreach (const Template &t, src) {
            QString id = t.file.get<QString>(key);
            if (!excluded.contains(id)) {
                dst.append(t);
                excluded.insert(id);
            }
        }
    }

public:
    FilterDupeMetadataTransform() : TimeVaryingTransform(false,false) {}
};

BR_REGISTER(Transform, FilterDupeMetadataTransform)

} // namespace br

#include "metadata/filterdupemetadata.moc"
