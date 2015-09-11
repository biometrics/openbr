/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2015 Rank One Computing Corporation                             *
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
 * \brief Filters templates such that the only remaining template wills have metadata values witihin the
 *          the specified ranges. 
 * \br_property QString key1 The meta-data key(s) to filter on
 * \br_property float value1 The values to compare the values of key1 and key2 entires against
 * \br_property QString compareType1 The comparison operation to perform. "le" ->  val(key1) <= value1, "lt" -> val(key1) < value1,
 *                      "ge" -> val(key1) >= value1, "gt" -> val(key1) > value1, "eq" -> val(key) == value1.
 * \author Brendan Klare \cite bklare
 */
class FilterMetadataTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString key1 READ get_key1 WRITE set_key1 RESET reset_key1 STORED false)
    Q_PROPERTY(QString compareType1 READ get_compareType1 WRITE set_compareType1 RESET reset_compareType1 STORED false)
    Q_PROPERTY(float value1 READ get_value1 WRITE set_value1 RESET reset_value1 STORED false)
    BR_PROPERTY(QString, key1, "")
    BR_PROPERTY(QString, compareType1, "")
    BR_PROPERTY(float, value1, 0)

    bool getCompare(float val1, float val2, QString compareType) const
    {
        bool pass = false;
        if (compareType == "gt")
            pass = val1 > val2;
        else if (compareType == "ge")
            pass = val1 >= val2;
        else if (compareType == "lt")
            pass = val1 < val2;
        else if (compareType == "le")
            pass = val1 <= val2;
        else if (compareType == "eq")
            pass = val1 == val2;
        else
            qDebug() << "Unknown compare type: " << compareType;
        return pass;
    }

    void project(const Template &, Template &) const
    {
        qFatal("Filter can't do anything here");
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach (const Template &srcTemp, src) 
        {
            if (!srcTemp.file.contains(key1))
                continue;
            
            if (getCompare(srcTemp.file.get<float>(key1), value1, compareType1))
                dst.append(srcTemp);
        }
    }
            
};

BR_REGISTER(Transform, FilterMetadataTransform)

} // namespace br

#include "metadata/filtermetadata.moc"
