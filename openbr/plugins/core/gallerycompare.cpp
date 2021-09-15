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
 * \ingroup transforms
 * \brief Compare each Template to a fixed Gallery (with name = galleryName), using the specified distance.
 * dst will contain a 1 by n vector of scores.
 * \author Charles Otto \cite caotto
 */
class GalleryCompareTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Distance *distance READ get_distance WRITE set_distance RESET reset_distance STORED true)
    Q_PROPERTY(QString galleryName READ get_galleryName WRITE set_galleryName RESET reset_galleryName STORED false)
    BR_PROPERTY(br::Distance*, distance, NULL)
    BR_PROPERTY(QString, galleryName, "")

    TemplateList gallery;

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        if (gallery.isEmpty())
            return;

        QList<float> line = distance->compare(gallery, src);
        dst.m() = OpenCVUtils::toMat(line, 1);
    }

    void init()
    {
        if (!galleryName.isEmpty())
            gallery = TemplateList::fromGallery(galleryName);
    }

    void train(const TemplateList &data)
    {
        gallery = data;
    }

    void store(QDataStream &stream) const
    {
        br::Object::store(stream);
        stream << gallery;
    }

    void load(QDataStream &stream)
    {
        br::Object::load(stream);
        stream >> gallery;
    }

public:
    GalleryCompareTransform() : Transform(false, true) {}
};

BR_REGISTER(Transform, GalleryCompareTransform)

} // namespace br

#include "core/gallerycompare.moc"
