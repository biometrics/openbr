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

#include <QtConcurrent>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

static void _train(Transform *transform, const QList<TemplateList> *data)
{
    transform->train(*data);
}

/*!
 * \ingroup transforms
 * \brief Transforms in parallel.
 *
 * The source Template is seperately given to each transform and the results are appended together.
 *
 * \author Josh Klontz \cite jklontz
 * \br_related_plugin PipeTransform
 */
class ForkTransform : public CompositeTransform
{
    Q_OBJECT

    void train(const QList<TemplateList> &data)
    {
        if (!trainable) return;
        QFutureSynchronizer<void> futures;
        for (int i=0; i<transforms.size(); i++)
            futures.addFuture(QtConcurrent::run(_train, transforms[i], &data));
        futures.waitForFinished();
    }

    // same as _project, but calls projectUpdate on sub-transforms
    void projectupdate(const Template &src, Template &dst)
    {
        foreach (Transform *f, transforms) {
            try {
                Template res;
                f->projectUpdate(src, res);
                dst.merge(res);
            } catch (...) {
                qWarning("Exception triggered when processing %s with transform %s", qPrintable(src.file.flat()), qPrintable(f->objectName()));
                dst = Template(src.file);
                dst.file.fte = true;
                break;
            }
        }
    }

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst.reserve(src.size());
        for (int i=0; i<src.size(); i++) dst.append(Template(src[i].file));
        foreach (Transform *f, transforms) {
            TemplateList m;
            f->projectUpdate(src, m);
            if (m.size() != dst.size()) qFatal("TemplateList is of an unexpected size.");
            for (int i=0; i<src.size(); i++) dst[i].merge(m[i]);
        }
    }

    // this is probably going to go bad, fork transform probably won't work well in a variable
    // input/output scenario
    virtual void finalize(TemplateList &output)
    {
        output.clear();
        // For each transform,
        for (int i = 0; i < transforms.size(); i++)
        {
            // Collect any final templates
            TemplateList last_set;
            transforms[i]->finalize(last_set);
            if (last_set.empty())
                continue;

            if (output.empty()) output = last_set;
            else
            {
                // is the number of templates received from this transform consistent with the number
                // received previously? If not we can't do anything coherent here.
                if (last_set.size() != output.size())
                    qFatal("mismatched template list sizes in ForkTransform");
                for (int j = 0; j < output.size(); j++) {
                    output[j].append(last_set[j]);
                }
            }
        }
    }

protected:

    // Apply each transform to src, concatenate the results
    void _project(const Template &src, Template &dst) const
    {
        foreach (const Transform *f, transforms) {
            try {
                dst.merge((*f)(src));
            } catch (...) {
                qWarning("Exception triggered when processing %s with transform %s", qPrintable(src.file.flat()), qPrintable(f->objectName()));
                dst = Template(src.file);
                dst.file.fte = true;
                break;
            }
        }
    }

    void _project(const TemplateList &src, TemplateList &dst) const
    {
        dst.reserve(src.size());
        for (int i=0; i<src.size(); i++) dst.append(Template(src[i].file));
        foreach (const Transform *f, transforms) {
            TemplateList m;
            f->project(src, m);
            if (m.size() != dst.size()) qFatal("TemplateList is of an unexpected size.");
            for (int i=0; i<src.size(); i++) dst[i].merge(m[i]);
        }
    }

};

BR_REGISTER(Transform, ForkTransform)

} // namespace br

#include "core/fork.moc"
