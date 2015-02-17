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
 * \brief Attenuation function based distance from attributes
 * \author Scott Klum \cite sklum
 */
class AttributeDistance : public UntrainableDistance
{
    Q_OBJECT
    Q_PROPERTY(QString attribute READ get_attribute WRITE set_attribute RESET reset_attribute STORED false)
    BR_PROPERTY(QString, attribute, QString())

    float compare(const Template &target, const Template &query) const
    {
        float queryValue = query.file.get<float>(attribute);
        float targetValue = target.file.get<float>(attribute);

        // TODO: Set this magic number to something meaningful
        float stddev = 1;

        if (queryValue == targetValue) return 1;
        else return 1/(stddev*sqrt(2*CV_PI))*exp(-0.5*pow((targetValue-queryValue)/stddev, 2));
    }
};

BR_REGISTER(Distance, AttributeDistance)

} // namespace br

#include "distance/attribute.moc"
