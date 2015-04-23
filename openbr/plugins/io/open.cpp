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

#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Applies Format to Template filename and appends results.
 * \author Josh Klontz \cite jklontz
 */
class OpenTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        if (src.empty()) {
            if (Globals->verbose)
                qDebug("Opening %s", qPrintable(src.file.flat()));

            // Read from disk otherwise
            foreach (const File &file, src.file.split()) {
                QScopedPointer<Format> format(Factory<Format>::make(file));
                Template t = format->read();
                if (t.isEmpty())
                    qWarning("Can't open %s from %s", qPrintable(file.flat()), qPrintable(QDir::currentPath()));
                dst.append(t);
                dst.file.append(t.file.localMetadata());
            }
            if (dst.isEmpty())
                dst.file.fte = true;
        } else {
            // Propogate or decode existing matricies
            foreach (const Mat &m, src) {
                if (((m.rows > 1) && (m.cols > 1)) || (m.type() != CV_8UC1))
                    dst += m;
                else {
                    Mat dec = imdecode(src.m(), IMREAD_UNCHANGED);
                    if (dec.empty()) qWarning("Can't decode %s", qPrintable(src.file.flat()));
                    else dst += dec;
                }
            }
        }
    }
};

BR_REGISTER(Transform, OpenTransform)

} // namespace br

#include "io/open.moc"
