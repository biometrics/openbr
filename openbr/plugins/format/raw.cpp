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

using namespace cv;

namespace br
{

/*!
 * \ingroup formats
 * \brief RAW format
 *
 * \br_link http://www.nist.gov/srd/nistsd27.cfm
 * \author Josh Klontz \cite jklontz
 */
class rawFormat : public Format
{
    Q_OBJECT
    static QHash<QString, QHash<QString,QSize> > imageSizes; // QHash<Path, QHash<File,Size> >

    Template read() const
    {
        QString path = file.path();
        if (!imageSizes.contains(path)) {
            static QMutex mutex;
            QMutexLocker locker(&mutex);

            if (!imageSizes.contains(path)) {
                const QString imageSize = path+"/ImageSize.txt";
                QStringList lines;
                if (QFileInfo(imageSize).exists()) {
                    lines = QtUtils::readLines(imageSize);
                    lines.removeFirst(); // Remove header
                }

                QHash<QString,QSize> sizes;
                QRegExp whiteSpace("\\s+");
                foreach (const QString &line, lines) {
                    QStringList words = line.split(whiteSpace);
                    if (words.size() != 3) continue;
                    sizes.insert(words[0], QSize(words[2].toInt(), words[1].toInt()));
                }

                imageSizes.insert(path, sizes);
            }
        }

        QByteArray data;
        QtUtils::readFile(file, data);

        QSize size = imageSizes[path][file.baseName()];
        if (!size.isValid()) size = QSize(800,768);
        if (data.size() != size.width() * size.height())
            qFatal("Expected %d*%d bytes, got %d.", size.height(), size.width(), data.size());
        return Template(file, Mat(size.height(), size.width(), CV_8UC1, data.data()).clone());
    }

    void write(const Template &t) const
    {
        QtUtils::writeFile(file, QByteArray().setRawData((const char*)t.m().data, t.m().total() * t.m().elemSize()));
    }
};

QHash<QString, QHash<QString,QSize> > rawFormat::imageSizes;

BR_REGISTER(Format, rawFormat)

} // namespace br

#include "format/raw.moc"
