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

/*!
 * \ingroup galleries
 * \brief Treat the file as a single binary Template.
 * \author Josh Klontz \cite jklontz
 */
class templateGallery : public Gallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        *done = true;
        QByteArray data;
        QtUtils::readFile(file.name.left(file.name.size()-QString(".template").size()), data);
        return TemplateList() << Template(file, cv::Mat(1, data.size(), CV_8UC1, data.data()).clone());
    }

    void write(const Template &t)
    {
        (void) t;
        qFatal("Not supported.");
    }

    void init()
    {
        //
    }
};

BR_REGISTER(Gallery, templateGallery)

} // namespace br

#include "gallery/template.moc"
