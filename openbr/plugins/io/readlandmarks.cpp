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
 * \brief Read landmarks from a file and associate them with the correct Templates.
 * \author Scott Klum \cite sklum
 * \br_format Example of the format:
 *
 * image_001.jpg:146.000000,190.000000,227.000000,186.000000,202.000000,256.000000
 * image_002.jpg:75.000000,235.000000,140.000000,225.000000,91.000000,300.000000
 * image_003.jpg:158.000000,186.000000,246.000000,188.000000,208.000000,233.000000
 *
 */
class ReadLandmarksTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    Q_PROPERTY(QString file READ get_file WRITE set_file RESET reset_file STORED false)
    Q_PROPERTY(QString imageDelimiter READ get_imageDelimiter WRITE set_imageDelimiter RESET reset_imageDelimiter STORED false)
    Q_PROPERTY(QString landmarkDelimiter READ get_landmarkDelimiter WRITE set_landmarkDelimiter RESET reset_landmarkDelimiter STORED false)
    BR_PROPERTY(QString, file, QString())
    BR_PROPERTY(QString, imageDelimiter, ":")
    BR_PROPERTY(QString, landmarkDelimiter, ",")

    QHash<QString, QList<QPointF> > landmarks;

    void init()
    {
        if (file.isEmpty())
            return;

        QFile f(file);
        if (!f.open(QFile::ReadOnly | QFile::Text))
            qFatal("Failed to open %s for reading.", qPrintable(f.fileName()));

        while (!f.atEnd()) {
            const QStringList words = QString(f.readLine()).split(imageDelimiter);
            const QStringList lm = words[1].split(landmarkDelimiter);

            QList<QPointF> points;
            bool ok;
            for (int i=0; i<lm.size(); i+=2)
                points.append(QPointF(lm[i].toFloat(&ok),lm[i+1].toFloat(&ok)));
            if (!ok) qFatal("Failed to read landmark.");

            landmarks.insert(words[0],points);
        }
    }

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        dst.appendPoints(landmarks[dst.fileName()]);
    }
};

BR_REGISTER(Transform, ReadLandmarksTransform)

} // namespace br

#include "io/readlandmarks.moc"
