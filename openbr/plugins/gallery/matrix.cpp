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

namespace br
{

/*!
 * \ingroup galleries
 * \brief Combine all Templates into one large matrix and process it as a Format
 * \author Josh Klontz \cite jklontz
 */
class matrixGallery : public Gallery
{
    Q_OBJECT
    Q_PROPERTY(const QString extension READ get_extension WRITE set_extension RESET reset_extension STORED false)
    BR_PROPERTY(QString, extension, "mtx")

    TemplateList templates;

    ~matrixGallery()
    {
        if (templates.isEmpty())
            return;

        QScopedPointer<Format> format(Factory<Format>::make(getFormat()));
        format->write(Template(file, OpenCVUtils::toMat(templates.data())));
    }

    File getFormat() const
    {
        return file.name.left(file.name.size() - file.suffix().size()) + extension;
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        return TemplateList() << getFormat();
    }

    void write(const Template &t)
    {
        templates.append(t);
    }
};

BR_REGISTER(Gallery, matrixGallery)

} // namespace br

#include "gallery/matrix.moc"
