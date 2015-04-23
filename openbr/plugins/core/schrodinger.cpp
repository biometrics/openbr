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
 * \brief Generates two Templates, one of which is passed through a Transform and the other
 *        is not. No cats were harmed in the making of this Transform.
 * \author Scott Klum \cite sklum
 */
class SchrodingerTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform)
    BR_PROPERTY(br::Transform*, transform, NULL)

public:
    void train(const TemplateList &data)
    {
        transform->train(data);
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach(const Template &t, src) {
            dst.append(t);
            Template u;
            transform->project(t,u);
            dst.append(u);
        }
    }

    void project(const Template &src, Template &dst) const {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

};
BR_REGISTER(Transform, SchrodingerTransform)

} // namespace br

#include "core/schrodinger.moc"
