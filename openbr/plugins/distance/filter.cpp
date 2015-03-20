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
 * \brief Checks target metadata against filters.
 * \author Josh Klontz \cite jklontz
 */
class FilterDistance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        (void) b; // Query template isn't checked
        foreach (const QString &key, Globals->filters.keys()) {
            bool keep = false;
            const QString metadata = a.file.get<QString>(key, "");
            if (Globals->filters[key].isEmpty()) continue;
            if (metadata.isEmpty()) return -std::numeric_limits<float>::max();
            foreach (const QString &value, Globals->filters[key]) {
                if (metadata == value) {
                    keep = true;
                    break;
                }
            }
            if (!keep) return -std::numeric_limits<float>::max();
        }
        return 0;
    }
};

BR_REGISTER(Distance, FilterDistance)

} // namespace br

#include "distance/filter.moc"
