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
 * \brief Add any ground truth to the Template using the file's base name.
 * \author Josh Klontz \cite jklontz
 */
class GroundTruthTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString groundTruth READ get_groundTruth WRITE set_groundTruth RESET reset_groundTruth STORED false)
    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    BR_PROPERTY(QString, groundTruth, "")
    BR_PROPERTY(QStringList, keys, QStringList())

    QMap<QString,File> files;

    void init()
    {
        foreach (const File &file, TemplateList::fromGallery(groundTruth).files())
            files.insert(file.baseName(), file);
    }

    void projectMetadata(const File &src, File &dst) const
    {
        (void) src;
        foreach(const QString &key, keys)
            dst.set(key,files[dst.baseName()].value(key));
    }
};

BR_REGISTER(Transform, GroundTruthTransform)

} // namespace br

#include "metadata/groundtruth.moc"
