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
    Q_PROPERTY(QString imageName READ get_imageName WRITE set_imageName RESET reset_imageName STORED false)
    Q_PROPERTY(QString imgExtension READ get_imgExtension WRITE set_imgExtension RESET reset_imgExtension STORED false)
    BR_PROPERTY(QString, outputDirectory, "Temp")
    BR_PROPERTY(QString, imageName, "image")
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
        OpenCVUtils::saveImage(dst.m(), QString("%1/%2_%3.%4").arg(outputDirectory).arg(imageName).arg(cnt++, 5, 10, QChar('0')).arg(imgExtension));
    }

};

BR_REGISTER(Transform, WriteTransform)

} // namespace br

#include "io/write.moc"
