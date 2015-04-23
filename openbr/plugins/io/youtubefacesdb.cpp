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

#include <QCoreApplication>
#include <QProcess>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Implements the YouTubesFaceDB
 * \br_paper Wolf, Lior, Tal Hassner, and Itay Maoz.
 *           "Face recognition in unconstrained videos with matched background similarity."
 *           Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on. IEEE, 2011.
 * \author Josh Klontz \cite jklontz
 */
class YouTubeFacesDBTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString algorithm READ get_algorithm WRITE set_algorithm RESET reset_algorithm STORED false)
    BR_PROPERTY(QString, algorithm, "")

    void project(const Template &src, Template &dst) const
    {
        static QMutex mutex;

        // First input is the header in 'splits.txt'
        if (src.file.get<int>("Index") == 0) return;

        const QStringList words = src.file.name.split(", ");
        const QString matrix = "YTF-"+algorithm+"/"+words[0] + "_" + words[1] + "_" + words[4] + ".mtx";
        const QStringList arguments = QStringList() << "-algorithm" << algorithm
                                                    << "-parallelism" << QString::number(Globals->parallelism)
                                                    << "-path" << Globals->path
                                                    << "-compare" << File(words[2]).resolved() << File(words[3]).resolved() << matrix;
        mutex.lock();
        int result = 0;
        if (!QFileInfo(matrix).exists())
            result = QProcess::execute(QCoreApplication::applicationFilePath(), arguments);
        mutex.unlock();

        if (result != 0)
            qWarning("Process for computing %s returned %d.", qPrintable(matrix), result);
        dst = Template();
    }
};

BR_REGISTER(Transform, YouTubeFacesDBTransform)

} // namespace br

#include "io/youtubefacesdb.moc"
