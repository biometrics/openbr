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
 * \ingroup galleries
 * \brief Treats the gallery as a Format.
 * \author Josh Klontz \cite jklontz
 */
class DefaultGallery : public Gallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        *done = true;
        return TemplateList() << file;
    }

    void write(const Template &t)
    {
        QScopedPointer<Format> format(Factory<Format>::make(file));
        format->write(t);
    }

    qint64 totalSize()
    {
        return 1;
    }
};

BR_REGISTER(Gallery, DefaultGallery)

} //  namespace br

#include "gallery/default.moc"
