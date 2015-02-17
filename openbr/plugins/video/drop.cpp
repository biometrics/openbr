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
 * \brief Only use one frame every n frames.
 * \author Austin Blanton \cite imaus10
 *
 * For a video with m frames, DropFrames will pass on m/n frames.
 */
class DropFrames : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    BR_PROPERTY(int, n, 1)

    void project(const TemplateList &src, TemplateList &dst) const
    {
        if (src.first().file.get<int>("FrameNumber") % n != 0) return;
        dst = src;
    }

    void project(const Template &src, Template &dst) const
    {
        (void) src; (void) dst; qFatal("shouldn't be here");
    }
};

BR_REGISTER(Transform, DropFrames)

} // namespace br

#include "video/drop.moc"
