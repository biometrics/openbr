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
 * \brief Rename first found metadata key
 * \author Josh Klontz \cite jklontz
 */
class RenameFirstTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList find READ get_find WRITE set_find RESET reset_find STORED false)
    Q_PROPERTY(QString replace READ get_replace WRITE set_replace RESET reset_replace STORED false)
    BR_PROPERTY(QStringList, find, QStringList())
    BR_PROPERTY(QString, replace, "")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        foreach (const QString &key, find)
            if (dst.localKeys().contains(key)) {
                dst.set(replace, dst.value(key));
                dst.remove(key);
                break;
            }
    }
};

BR_REGISTER(Transform, RenameFirstTransform)

} // namespace br

#include "metadata/renamefirst.moc"
