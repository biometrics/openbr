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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Write all mats to disk as images.
 * \author Brendan Klare \cite bklare
 */
class WriteTransform : public TimeVaryingTransform
{
    Q_OBJECT
    Q_PROPERTY(QString outputDirectory READ get_outputDirectory WRITE set_outputDirectory RESET reset_outputDirectory STORED false)
    Q_PROPERTY(QString underscore READ get_underscore WRITE set_underscore RESET reset_underscore STORED false)
    Q_PROPERTY(QString imgExtension READ get_imgExtension WRITE set_imgExtension RESET reset_imgExtension STORED false)
    BR_PROPERTY(QString, outputDirectory, "Temp")
    BR_PROPERTY(QString, underscore, "")
    BR_PROPERTY(QString, imgExtension, "jpg")

    int cnt;

    void init() {
        cnt = 0;
        if (! QDir(outputDirectory).exists())
            QDir().mkdir(outputDirectory);
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        dst = src;
        QString path = QString("%1/%2%3.%4").arg(outputDirectory).arg(dst.file.baseName()).arg(underscore.isEmpty() ? "" : "_" + underscore).arg(imgExtension);
        OpenCVUtils::saveImage(dst.m(), path);
    }

};

BR_REGISTER(Transform, WriteTransform)

} // namespace br

#include "io/write.moc"
