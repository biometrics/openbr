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
 * \brief DOCUMENT ME CHARLES
 * \author Unknown \cite Unknown
 */
class FileExclusionTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(QString exclusionGallery READ get_exclusionGallery WRITE set_exclusionGallery RESET reset_exclusionGallery STORED false)
    BR_PROPERTY(QString, exclusionGallery, "")

    QSet<QString> excluded;

    void project(const Template &, Template &) const
    {
        qFatal("FileExclusion can't do anything here");
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach (const Template &srcTemp, src) {
            if (!excluded.contains(srcTemp.file))
                dst.append(srcTemp);
        }
    }

    void init()
    {
        if (exclusionGallery.isEmpty())
            return;
        File rFile(exclusionGallery);
        rFile.remove("append");

        FileList temp = FileList::fromGallery(rFile);
        excluded = QSet<QString>::fromList(temp.names());
    }
};

BR_REGISTER(Transform, FileExclusionTransform)

} // namespace br

#include "metadata/fileexclusion.moc"
