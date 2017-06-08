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

#include <QJsonObject>
#include <QJsonDocument>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Represent the metadata as JSON template data.
 * \author Josh Klontz \cite jklontz
 */
class JSONTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        dst.file.set("AlgorithmID", 2);
        QByteArray json = QJsonDocument(QJsonObject::fromVariantMap(dst.file.localMetadata())).toJson().replace('\n', ' ');
        dst += cv::Mat(1, json.size()+1 /*include null terminator*/, CV_8UC1, json.data()).clone();
    }
};

BR_REGISTER(Transform, JSONTransform)

} // namespace br

#include "metadata/json.moc"
