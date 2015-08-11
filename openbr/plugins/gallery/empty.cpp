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

#include <QtConcurrent>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>
#include <openbr/core/utility.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Reads/writes templates to/from folders.
 * \author Josh Klontz \cite jklontz
 * \br_property QString regexp An optional regular expression to match against the files extension.
 */
class EmptyGallery : public Gallery
{
    Q_OBJECT
    Q_PROPERTY(QString regexp READ get_regexp WRITE set_regexp RESET reset_regexp STORED false)
    BR_PROPERTY(QString, regexp, QString())

    qint64 gallerySize;

    void init()
    {
        QDir dir(file.name);
        QtUtils::touchDir(dir);
        QDirIterator it(dir.absolutePath(), QDir::Files | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
        gallerySize = 0;
        while (it.hasNext()) {
            it.next();
            gallerySize++;
        }
    }

    TemplateList readBlock(bool *done)
    {
        TemplateList templates;
        *done = true;

        // Enrolling a null file is used as an idiom to initialize an algorithm
        if (file.isNull()) return templates;

        // Add immediate subfolders
        QDir dir(file);
        QList< QFuture<TemplateList> > futures;
        foreach (const QString &folder, QtUtils::naturalSort(dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot))) {
            const QDir subdir = dir.absoluteFilePath(folder);
            futures.append(QtConcurrent::run(&EmptyGallery::getTemplates, subdir));
        }
        foreach (const QFuture<TemplateList> &future, futures)
            templates.append(future.result());

        // Add root folder
        foreach (const QString &fileName, getFiles(file.name, false))
            templates.append(File(fileName, dir.dirName()));

        if (!regexp.isEmpty()) {
            QRegExp re(regexp);
            re.setPatternSyntax(QRegExp::Wildcard);
            for (int i=templates.size()-1; i>=0; i--) {
                if (!re.exactMatch(templates[i].file.fileName())) {
                    templates.removeAt(i);
                }
            }
        }

        for (int i = 0; i < templates.size(); i++) templates[i].file.set("progress", i);

        return templates;
    }

    void write(const Template &t)
    {
        static QMutex diskLock;

        // Enrolling a null file is used as an idiom to initialize an algorithm
        if (file.name.isEmpty()) return;

        const QString newFormat = file.get<QString>("newFormat",QString());
        QString destination = file.name + "/" + (file.getBool("preservePath") ? t.file.path()+"/" : QString());
        destination += (newFormat.isEmpty() ? t.file.fileName() : t.file.baseName()+newFormat);

        QMutexLocker diskLocker(&diskLock); // Windows prefers to crash when writing to disk in parallel
        if (t.isNull()) {
            QtUtils::copyFile(t.file.resolved(), destination);
        } else {
            QScopedPointer<Format> format(Factory<Format>::make(destination));
            format->write(t);
        }
    }

    qint64 totalSize()
    {
        return gallerySize;
    }

    static TemplateList getTemplates(const QDir &dir)
    {
        const QStringList files = getFiles(dir, true);
        TemplateList templates; templates.reserve(files.size());
        foreach (const QString &file, files)
            templates.append(File(file, dir.dirName()));
        return templates;
    }
};

BR_REGISTER(Gallery, EmptyGallery)

} // namespace br

#include "gallery/empty.moc"
