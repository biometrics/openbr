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
class KeyToLandmarkTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    Q_PROPERTY(bool point READ get_point WRITE set_point RESET reset_point STORED false)
    BR_PROPERTY(QString, key, "")
    BR_PROPERTY(bool, point, false)

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        if (point) {
            if (src.contains(QStringList() << key + "_X" << key + "_Y"))
                dst.appendPoint(QPointF(src.get<float>(key + "_X"),
                                        src.get<float>(key + "_Y")));
        } else {
            if (src.contains(QStringList() << key + "_X" << key + "_Y" << key + "_Width" << key + "_Height"))
                dst.appendRect(QRectF(src.get<float>(key + "_X"),
                                      src.get<float>(key + "_Y"),
                                      src.get<float>(key + "_Width"),
                                      src.get<float>(key + "_Height")));
        }

    }
};

BR_REGISTER(Transform, KeyToLandmarkTransform)

} // namespace br

#include "metadata/keytolandmark.moc"
