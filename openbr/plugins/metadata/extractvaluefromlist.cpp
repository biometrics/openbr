
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
 * \brief Extracts a single value from a list and sets the specified key to that value.
 * \br_property QString key The meta-data key that contains a list of values
 * \br_property int index The index of the value of interest (0 offset)
 * \br_property QString outputKey The output metadata key which stores the value of interest
 * \author Keyur Patel \cite kpatel
 */
class ExtractValueFromListTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    Q_PROPERTY(int index READ get_index WRITE set_index RESET reset_index STORED false)
    Q_PROPERTY(QString outputKey READ get_outputKey WRITE set_outputKey RESET reset_outputKey STORED false)

    BR_PROPERTY(QString, key, "")
    BR_PROPERTY(int, index, 0)
    BR_PROPERTY(QString, outputKey, "ExtractedValue")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        if (src.contains(key)){
            QList<float> values = src.getList<float>(key);
            if (values.size() -1 >= index && index >= 0)
                dst.set(outputKey,values[index]);
        } 
    }

};

BR_REGISTER(Transform, ExtractValueFromListTransform)

} // namespace br

#include "metadata/extractvaluefromlist.moc"

