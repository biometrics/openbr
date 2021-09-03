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
 * \brief DOCUMENT ME
 * \author Unknown \cite unknown
 */
class DiscardTemplatesTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        (void) src; (void) dst;
        qFatal("Incorrect project called on DiscardTemplatesTransform");
    }
    void project(const TemplateList &src, TemplateList &dst) const
    {
        (void) src;
        dst.clear();
    }
};

BR_REGISTER(Transform, DiscardTemplatesTransform)

} // namespace br

#include "core/discardtemplates.moc"
