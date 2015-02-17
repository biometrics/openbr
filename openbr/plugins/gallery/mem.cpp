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
 * \ingroup initializers
 * \brief Initialization support for memGallery.
 * \author Josh Klontz \cite jklontz
 */
class MemoryGalleries : public Initializer
{
    Q_OBJECT

    void initialize() const {}

    void finalize() const
    {
        galleries.clear();
    }

public:
    static QHash<File, TemplateList> galleries; /*!< TODO */
};

QHash<File, TemplateList> MemoryGalleries::galleries;

BR_REGISTER(Initializer, MemoryGalleries)

/*!
 * \ingroup galleries
 * \brief A gallery held in memory.
 * \author Josh Klontz \cite jklontz
 */
class memGallery : public Gallery
{
    Q_OBJECT
    int block;
    qint64 gallerySize;

    void init()
    {
        block = 0;
        File galleryFile = file.name.mid(0, file.name.size()-4);
        if ((galleryFile.suffix() == "gal") && galleryFile.exists() && !MemoryGalleries::galleries.contains(file)) {
            QSharedPointer<Gallery> gallery(Factory<Gallery>::make(galleryFile));
            MemoryGalleries::galleries[file] = gallery->read();
            gallerySize = MemoryGalleries::galleries[file].size();
        }
    }

    TemplateList readBlock(bool *done)
    {
        TemplateList templates = MemoryGalleries::galleries[file].mid(block*readBlockSize, readBlockSize);
        for (qint64 i = 0; i < templates.size();i++) {
            templates[i].file.set("progress", i + block * readBlockSize);
        }

        *done = (templates.size() < readBlockSize);
        block = *done ? 0 : block+1;
        return templates;
    }

    void write(const Template &t)
    {
        MemoryGalleries::galleries[file].append(t);
    }

    qint64 totalSize()
    {
        return gallerySize;
    }

    qint64 position()
    {
        return block * readBlockSize;
    }

};

BR_REGISTER(Gallery, memGallery)

FileList FileList::fromGallery(const File &rFile, bool cache)
{
    File file = rFile;
    file.remove("append");

    File targetMeta = file;
    targetMeta.name = targetMeta.path() + targetMeta.baseName() + "_meta" + targetMeta.hash() + ".mem";

    FileList fileData;

    // Did we already read the data?
    if (MemoryGalleries::galleries.contains(targetMeta))
    {
        return MemoryGalleries::galleries[targetMeta].files();
    }

    TemplateList templates;
    // OK we read the data in some form, does the gallery type containing matrices?
    if ((QStringList() << "gal" << "mem" << "template" << "ut").contains(file.suffix())) {
        // Retrieve it block by block, dropping matrices from read templates.
        QScopedPointer<Gallery> gallery(Gallery::make(file));
        gallery->set_readBlockSize(10);
        bool done = false;
        while (!done)
        {
            TemplateList tList = gallery->readBlock(&done);
            for (int i=0; i < tList.size();i++)
            {
                tList[i].clear();
                templates.append(tList[i].file);
            }
        }
    }
    else {
        // this is a gallery format that doesn't include matrices, so we can just read it
        QScopedPointer<Gallery> gallery(Gallery::make(file));
        templates= gallery->read();
    }

    if (cache)
    {
        QScopedPointer<Gallery> memOutput(Gallery::make(targetMeta));
        memOutput->writeBlock(templates);
    }
    fileData = templates.files();
    return fileData;
}

} // namespace br

#include "gallery/mem.moc"
