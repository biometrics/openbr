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

#include <QStandardPaths>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Crawl a root location for image files.
 * \author Josh Klontz \cite jklontz
 */
class crawlGallery : public Gallery
{
    Q_OBJECT
    Q_PROPERTY(bool autoRoot READ get_autoRoot WRITE set_autoRoot RESET reset_autoRoot STORED false)
    Q_PROPERTY(int depth READ get_depth WRITE set_depth RESET reset_depth STORED false)
    Q_PROPERTY(bool depthFirst READ get_depthFirst WRITE set_depthFirst RESET reset_depthFirst STORED false)
    Q_PROPERTY(int images READ get_images WRITE set_images RESET reset_images STORED false)
    Q_PROPERTY(bool json READ get_json WRITE set_json RESET reset_json STORED false)
    Q_PROPERTY(int timeLimit READ get_timeLimit WRITE set_timeLimit RESET reset_timeLimit STORED false)
    BR_PROPERTY(bool, autoRoot, false)
    BR_PROPERTY(int, depth, INT_MAX)
    BR_PROPERTY(bool, depthFirst, false)
    BR_PROPERTY(int, images, INT_MAX)
    BR_PROPERTY(bool, json, false)
    BR_PROPERTY(int, timeLimit, INT_MAX)

    QTime elapsed;
    TemplateList templates;

    void crawl(QFileInfo url, int currentDepth = 0)
    {
        if ((templates.size() >= images) || (currentDepth >= depth) || (elapsed.elapsed()/1000 >= timeLimit))
            return;

        if (url.filePath().startsWith("file://"))
            url = QFileInfo(url.filePath().mid(7));

        if (url.isDir()) {
            const QDir dir(url.absoluteFilePath());
            const QFileInfoList files = dir.entryInfoList(QDir::Files);
            const QFileInfoList subdirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
            foreach (const QFileInfo &first, depthFirst ? subdirs : files)
                crawl(first, currentDepth + 1);
            foreach (const QFileInfo &second, depthFirst ? files : subdirs)
                crawl(second, currentDepth + 1);
        } else if (url.isFile()) {
            const QString suffix = url.suffix();
            if ((suffix == "bmp") || (suffix == "jpg") || (suffix == "jpeg") || (suffix == "png") || (suffix == "tiff")) {
                File f;
                if (json) f.set("URL", "file://"+url.canonicalFilePath());
                else      f.name = "file://"+url.canonicalFilePath();
                templates.append(f);
            }
        }
    }

    void init()
    {
        elapsed.start();
        const QString root = file.name.mid(0, file.name.size()-6); // Remove .crawl suffix";
        if (!root.isEmpty()) {
            crawl(root);
        } else {
            if (autoRoot) {
                foreach (const QString &path, QStandardPaths::standardLocations(QStandardPaths::HomeLocation))
                    crawl(path);
            } else {
                QFile file;
                file.open(stdin, QFile::ReadOnly);
                while (!file.atEnd()) {
                    const QString url = QString::fromLocal8Bit(file.readLine()).simplified();
                    if (!url.isEmpty())
                        crawl(url);
                }
            }
        }
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        return templates;
    }

    void write(const Template &)
    {
        qFatal("Not supported");
    }
};

BR_REGISTER(Gallery, crawlGallery)

} // namespace br

#include "gallery/crawl.moc"
