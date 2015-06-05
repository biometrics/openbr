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
 * \brief It's like the opposite of ExpandTransform, but not really
 *
 * Given a TemplateList as input, concatenate them into a single Template
 *
 * \author Charles Otto \cite caotto
 */
class ContractTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    virtual void project(const TemplateList &src, TemplateList &dst) const
    {
        if (src.empty()) return;
        Template out;

        foreach (const Template &t, src) {
            out.merge(t);
        }
        out.file.clearRects();
        foreach (const Template &t, src) {
            if (!t.file.rects().empty())
                out.file.appendRects(t.file.rects());
        }
        dst.clear();
        dst.append(out);
    }

    virtual void project(const Template &src, Template &dst) const
    {
        qFatal("this has gone bad");
        (void) src; (void) dst;
    }
};

BR_REGISTER(Transform, ContractTransform)

} // namespace br

#include "core/contract.moc"
