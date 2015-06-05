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
 * \brief Removes all but the first matrix from the Template.
 * \br_related_plugin IdentityTransform DiscardTransform RestTransform RemoveTransform
 * \author Josh Klontz \cite jklontz
 */
class FirstTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        // AggregateFrames will leave the Template empty
        // if it hasn't filled up the buffer
        // so we gotta anticipate an empty Template
        if (src.empty()) return;
        dst.file = src.file;
        dst = src.m();
    }
};

BR_REGISTER(Transform, FirstTransform)

} // namespace br

#include "core/first.moc"
