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
 * \brief Remove a name from a point/rect
 * \author Scott Klum \cite sklum
 */
class AnonymizeLandmarksTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList names READ get_names WRITE set_names RESET reset_names STORED false)
    BR_PROPERTY(QStringList, names, QStringList())

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        foreach (const QString &name, names) {
            if (src.contains(name)) {
                QVariant variant = src.value(name);
                if (variant.canConvert(QMetaType::QPointF)) {
                    dst.appendPoint(variant.toPointF());
                } else if (variant.canConvert(QMetaType::QRectF)) {
                    dst.appendRect(variant.toRectF());
                } else {
                    qFatal("Cannot convert landmark to point or rect.");
                }
            }
        }
    }
};

BR_REGISTER(Transform, AnonymizeLandmarksTransform)

} // namespace br

#include "metadata/anonymizelandmarks.moc"
