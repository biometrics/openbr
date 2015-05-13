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
 * \brief DOCUMENT ME CHARLES
 * \author Unknown \cite unknown
 */
class GalleryOutputTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString outputString READ get_outputString WRITE set_outputString RESET reset_outputString STORED false)
    BR_PROPERTY(QString, outputString, "")

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        if (src.empty())
            return;
        dst = src;
        for (int i=0; i < dst.size();i++) {
            if (dst[i].file.getBool("FTE"))
                dst[i].file.fte = true;
        }
        writer->writeBlock(dst);
    }

    void train(const TemplateList& data)
    {
        (void) data;
    }
    ;
    void init()
    {
        writer = QSharedPointer<Gallery>(Gallery::make(outputString));
    }

    QSharedPointer<Gallery> writer;
public:
    GalleryOutputTransform() : TimeVaryingTransform(false,false) {}
};

BR_REGISTER(Transform, GalleryOutputTransform)

} // namespace br

#include "io/galleryoutput.moc"
