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
#include <openbr/core/common.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Flags images that failed to enroll based on the specified Transform.
 * \author Josh Klontz \cite jklontz
 */
class FTETransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform)
    Q_PROPERTY(float min READ get_min WRITE set_min RESET reset_min)
    Q_PROPERTY(float max READ get_max WRITE set_max RESET reset_max)
    BR_PROPERTY(br::Transform*, transform, NULL)
    BR_PROPERTY(float, min, -std::numeric_limits<float>::max())
    BR_PROPERTY(float, max,  std::numeric_limits<float>::max())

    void train(const TemplateList &data)
    {
        transform->train(data);

        TemplateList projectedData;
        transform->project(data, projectedData);

        QList<float> vals;
        foreach (const Template &t, projectedData) {
            if (!t.file.contains(transform->objectName()))
                qFatal("Matrix metadata missing key %s.", qPrintable(transform->objectName()));
            vals.append(t.file.get<float>(transform->objectName()));
        }
        float q1, q3;
        Common::Median(vals, &q1, &q3);
        min = q1 - 1.5 * (q3 - q1);
        max = q3 + 1.5 * (q3 - q1);
    }

    void project(const Template &src, Template &dst) const
    {
        Template projectedSrc;
        transform->project(src, projectedSrc);
        const float val = projectedSrc.file.get<float>(transform->objectName());

        dst = src;
        dst.file.set(transform->objectName(), val);
        dst.file.set("PossibleFTE", (val < min) || (val > max));
    }
};

BR_REGISTER(Transform, FTETransform)

} // namespace br

#include "core/fte.moc"
