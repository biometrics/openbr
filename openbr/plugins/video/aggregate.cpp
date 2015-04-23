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
 * \brief Passes along n sequential frames to the next Transform.
 *
 * For a video with m frames, AggregateFrames would create a total of m-n+1 sequences ([0,n] ... [m-n+1, m])
 *
 * \author Josh Klontz \cite jklontz
 */
class AggregateFrames : public TimeVaryingTransform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    BR_PROPERTY(int, n, 1)

    TemplateList buffer;

public:
    AggregateFrames() : TimeVaryingTransform(false, false) {}

private:
    void train(const TemplateList &data)
    {
        (void) data;
    }

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        buffer.append(src);
        if (buffer.size() < n) return;
        Template out;
        foreach (const Template &t, buffer) out.append(t);
        out.file = buffer.takeFirst().file;
        dst.append(out);
    }

    void finalize(TemplateList &output)
    {
        (void) output;
        buffer.clear();
    }

    void store(QDataStream &stream) const
    {
        (void) stream;
    }

    void load(QDataStream &stream)
    {
        (void) stream;
    }
};

BR_REGISTER(Transform, AggregateFrames)

} // namespace br

#include "video/aggregate.moc"
