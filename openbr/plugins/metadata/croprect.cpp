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
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Crops the width and height of a Template's rects by input width and height factors.
 * \author Scott Klum \cite sklum
 */
class CropRectTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    Q_PROPERTY(QString widthCrop READ get_widthCrop WRITE set_widthCrop RESET reset_widthCrop STORED false)
    Q_PROPERTY(QString heightCrop READ get_heightCrop WRITE set_heightCrop RESET reset_heightCrop STORED false)
    BR_PROPERTY(QString, widthCrop, QString())
    BR_PROPERTY(QString, heightCrop, QString())

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        QList<QRectF> rects = src.rects();
        for (int i=0;i < rects.size(); i++) {
            rects[i].setX(rects[i].x() + rects[i].width() * QtUtils::toPoint(widthCrop).x());
            rects[i].setY(rects[i].y() + rects[i].height() * QtUtils::toPoint(heightCrop).x());
            rects[i].setWidth(rects[i].width() * (1-QtUtils::toPoint(widthCrop).y()));
            rects[i].setHeight(rects[i].height() * (1-QtUtils::toPoint(heightCrop).y()));
        }
        dst.setRects(rects);
    }
};

BR_REGISTER(Transform, CropRectTransform)

} // namespace br

#include "metadata/croprect.moc"
