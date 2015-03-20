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
 * \brief Convert values of key_X, key_Y, key_Width, key_Height to a rect.
 * \author Jordan Cheney \cite JordanCheney
 */
class KeyToRectTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    BR_PROPERTY(QString, key, "")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        if (src.contains(QStringList() << key + "_X" << key + "_Y" << key + "_Width" << key + "_Height"))
            dst.appendRect(QRectF(src.get<int>(key + "_X"),
                                  src.get<int>(key + "_Y"),
                                  src.get<int>(key + "_Width"),
                                  src.get<int>(key + "_Height")));

    }

};

BR_REGISTER(Transform, KeyToRectTransform)

} // namespace br

#include "metadata/keytorect.moc"
