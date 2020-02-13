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
 * \brief Rename metadata key
 * \author Josh Klontz \cite jklontz
 */
class RenameTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString find READ get_find WRITE set_find RESET reset_find STORED false)
    Q_PROPERTY(QString replace READ get_replace WRITE set_replace RESET reset_replace STORED false)
    Q_PROPERTY(bool keepOldKey READ get_keepOldKey WRITE set_keepOldKey RESET reset_keepOldKey STORED false)
    BR_PROPERTY(QString, find, "")
    BR_PROPERTY(QString, replace, "")
    BR_PROPERTY(bool, keepOldKey, false)

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        if (dst.localKeys().contains(find)) {
            if (replace == "_Points")
                dst.setPoints(dst.getList<QPointF>(find));
            else
                dst.set(replace, dst.value(find));
            if (!keepOldKey)
               dst.remove(find);
        }
    }
};

BR_REGISTER(Transform, RenameTransform)

} // namespace br

#include "metadata/rename.moc"
