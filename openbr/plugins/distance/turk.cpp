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
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Unmaps Turk HITs to be compared against query mats
 * \author Scott Klum \cite sklum
 */
class TurkDistance : public UntrainableDistance
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key)
    Q_PROPERTY(QStringList values READ get_values WRITE set_values RESET reset_values STORED false)
    BR_PROPERTY(QString, key, QString())
    BR_PROPERTY(QStringList, values, QStringList())

    bool targetHuman;
    bool queryMachine;

    void init()
    {
        targetHuman = Globals->property("TurkTargetHuman").toBool();
        queryMachine = Globals->property("TurkQueryMachine").toBool();
    }

    cv::Mat getValues(const Template &t) const
    {
        QList<float> result;
        foreach (const QString &value, values)
            result.append(t.file.get<float>(key + "_" + value));
        return OpenCVUtils::toMat(result, 1);
    }

    float compare(const Template &target, const Template &query) const
    {
        const cv::Mat a = targetHuman ? getValues(target) : target.m();
        const cv::Mat b = queryMachine ? query.m() : getValues(query);
        return -norm(a, b, cv::NORM_L1);
    }
};

BR_REGISTER(Distance, TurkDistance)

} // namespace br

#include "distance/turk.moc"
