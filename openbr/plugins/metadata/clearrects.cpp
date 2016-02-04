/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2016 Rank One Computing Corporation.
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
 * \brief Clears the rects from a Template
 * \author Brendan Klare \cite bklare
 */
class ClearRectsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        dst.clearRects();
    }
};

BR_REGISTER(Transform, ClearRectsTransform)

} // namespace br

#include "metadata/clearrects.moc"
