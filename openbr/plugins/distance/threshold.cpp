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
 * \ingroup distances
 * \brief Applys a hard threshold to similarity values.
 * \author Scott Klum \cite sklum
 */
class ThresholdDistance : public Distance
{
    Q_OBJECT

    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance)
    Q_PROPERTY(float min READ get_min WRITE set_min RESET reset_min)
    Q_PROPERTY(float max READ get_max WRITE set_max RESET reset_max)
    BR_PROPERTY(br::Distance*, distance, NULL)
    BR_PROPERTY(float, min, -std::numeric_limits<float>::max())
    BR_PROPERTY(float, max, std::numeric_limits<float>::max())

    bool trainable()
    {
        return distance->trainable();
    }

    void train(const TemplateList &src)
    {
        distance->train(src);
    }

    float compare(const Template &a, const Template &b) const
    {
        float value = distance->compare(a,b);
        if (value < min) value = min;
        else if (value > max) value = max;
        return value;
    }
};

BR_REGISTER(Distance, ThresholdDistance)

} // namespace br

#include "distance/threshold.moc"

