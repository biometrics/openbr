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
 * \brief Create face bounding box from two eye locations.
 * \author Brendan Klare \cite bklare
 *
 * \br_property double widthPadding Specifies what percentage of the interpupliary distance (ipd) will be padded in both horizontal directions.
 * \br_property double verticalLocation specifies where vertically the eyes are within the bounding box (0.5 would be the center).
 */
class FaceFromEyesTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(double widthPadding READ get_widthPadding WRITE set_widthPadding RESET reset_widthPadding STORED false)
    Q_PROPERTY(double verticalLocation READ get_verticalLocation WRITE set_verticalLocation RESET reset_verticalLocation STORED false)
    Q_PROPERTY(int leftEyeIdx READ get_leftEyeIdx WRITE set_leftEyeIdx RESET reset_leftEyeIdx STORED false)
    Q_PROPERTY(int rightEyeIdx READ get_rightEyeIdx WRITE set_rightEyeIdx RESET reset_rightEyeIdx STORED false)
    BR_PROPERTY(double, widthPadding, 0.7)
    BR_PROPERTY(double, verticalLocation, 0.25)
    BR_PROPERTY(int, leftEyeIdx, 0)
    BR_PROPERTY(int, rightEyeIdx, 1)

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        if (src.points().isEmpty()) {
            qWarning("No landmarks");
            return;
        }

        QPointF eyeL = src.points()[leftEyeIdx];
        QPointF eyeR = src.points()[rightEyeIdx];
        QPointF eyeCenter((eyeL.x() + eyeR.x()) / 2, (eyeL.y() + eyeR.y()) / 2);
        float ipd = sqrt(pow(eyeL.x() - eyeR.x(), 2) + pow(eyeL.y() - eyeR.y(), 2));
        float width = ipd + 2 * widthPadding * ipd;

        dst.appendRect(QRectF(eyeCenter.x() - width / 2, eyeCenter.y() - width * verticalLocation, width, width));
    }
};

BR_REGISTER(Transform, FaceFromEyesTransform)

} // namespace br

#include "metadata/facefromeyes.moc"
