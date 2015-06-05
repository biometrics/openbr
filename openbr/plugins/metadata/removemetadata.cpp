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
 * \brief Removes a metadata field from all Templates
 * \author Brendan Klare \cite bklare
 */
class RemoveMetadataTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString attributeName READ get_attributeName WRITE set_attributeName RESET reset_attributeName STORED false)
    BR_PROPERTY(QString, attributeName, "None")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        if (dst.contains(attributeName))
            dst.remove(attributeName);
    }
};

BR_REGISTER(Transform, RemoveMetadataTransform)

} // namespace br

#include "metadata/removemetadata.moc"
