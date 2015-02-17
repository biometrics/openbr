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

void FileGallery::init()
{
    f.setFileName(file);

    Gallery::init();
}

void FileGallery::writeOpen()
{
    if (!f.isOpen() ) {
        QtUtils::touchDir(f);
        if (!f.open(QFile::WriteOnly))
            qFatal("Failed to open %s for writing.", qPrintable(file));
    }
}

bool FileGallery::readOpen()
{
    if (!f.isOpen() ) {
        if (!f.exists() ) {
            qFatal("File %s does not exist.", qPrintable(file));
        }

        if (!f.open(QFile::ReadOnly))
            qFatal("Failed to open %s for reading.", qPrintable(file));
        return true;
    }
    return false;
}

qint64 FileGallery::totalSize()
{
    readOpen();
    return f.size();
}

} // namespace br
