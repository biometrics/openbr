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

/*!
 * \ingroup Transforms
 * \brief Transforms in series.
 *
 * The source Template is given to the first transform and the resulting Template is passed to the next transform, etc.
 *
 * \author Josh Klontz \cite jklontz
 * \br_related_plugin ExpandTransform ForkTransform
 */
class PipeTransform : public CompositeTransform
{
    Q_OBJECT

    void _projectPartial(TemplateList *srcdst, int startIndex, int stopIndex)
    {
        TemplateList ftes;
        for (int i=startIndex; i<stopIndex; i++) {
            TemplateList res;
            transforms[i]->project(*srcdst, res);

            splitFTEs(res, ftes);
            *srcdst = res;
        }
    }

    void train(const QList<TemplateList> &data)
    {
        if (!trainable) return;

        QList<TemplateList> dataLines(data);

        int i = 0;
        while (i < transforms.size()) {
            // Conditional statement covers likely case that first transform is untrainable
            if (transforms[i]->trainable) {
                qDebug() << "Training" << transforms[i]->description() << "\n...";
                transforms[i]->train(dataLines);
            }

            // if the transform is time varying, we can't project it in parallel
            if (transforms[i]->timeVarying()) {
                qDebug() << "Projecting" << transforms[i]->description() << "\n...";
                for (int j=0; j < dataLines.size();j++) {
                    TemplateList junk;
                    splitFTEs(dataLines[j], junk);

                    transforms[i]->projectUpdate(dataLines[j], dataLines[j]);
                }

                // advance i since we already projected for this stage.
                i++;

                // the next stage might be trainable, so continue to evaluate it.
                continue;
            }

            // We project through any subsequent untrainable transforms at once
            //   as a memory optimization in case any of these intermediate
            //   transforms allocate a lot of memory (like OpenTransform)
            //   then we don't want all the training templates to be processed
            //   by that transform at once if we can avoid it.
            int nextTrainableTransform = i+1;
            while ((nextTrainableTransform < transforms.size()) &&
                   !transforms[nextTrainableTransform]->trainable &&
                   !transforms[nextTrainableTransform]->timeVarying())
                nextTrainableTransform++;

            // No more trainable transforms? Don't need any more projects then
            if (nextTrainableTransform == transforms.size())
                break;

            fprintf(stderr, "Projecting %s", qPrintable(transforms[i]->description()));
            for (int j=i+1; j < nextTrainableTransform; j++)
                fprintf(stderr,"+%s", qPrintable(transforms[j]->description()));
            fprintf(stderr, "\n...\n");
            fflush(stderr);

            QFutureSynchronizer<void> futures;
            for (int j=0; j < dataLines.size(); j++)
                futures.addFuture(QtConcurrent::run(this, &PipeTransform::_projectPartial, &dataLines[j], i, nextTrainableTransform));
            futures.waitForFinished();

            i = nextTrainableTransform;
        }
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        dst = src;
        foreach (Transform *f, transforms) {
            try {
                f->projectUpdate(dst);
                if (dst.file.fte)
                    break;
            } catch (...) {
                qWarning("Exception triggered when processing %s with transform %s", qPrintable(src.file.flat()), qPrintable(f->objectName()));
                dst = Template(src.file);
                dst.file.fte = true;
                break;
            }
        }
    }

    // For time varying transforms, parallel execution over individual templates
    // won't work.
    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        TemplateList ftes;
        dst = src;
        foreach (Transform *f, transforms) {
            TemplateList res;
            f->projectUpdate(dst, res);
            splitFTEs(res, ftes);
            dst = res;
        }
        dst.append(ftes);
    }

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
            // Push any templates received through the remaining transforms in the sequence
            for (int j = (i+1); j < transforms.size();j++)
            {
                transforms[j]->projectUpdate(last_set);
            }
            // append the result to the output set
            output.append(last_set);
        }
    }

    void init()
    {
        QList<Transform *> flattened;
        for (int i=0;i < transforms.size(); i++)
        {
            PipeTransform *probe = dynamic_cast<PipeTransform *> (transforms[i]);
            if (!probe) {
                flattened.append(transforms[i]);
                continue;
            }
            for (int j=0; j < probe->transforms.size(); j++)
                flattened.append(probe->transforms[j]);
        }
        transforms = flattened;

        CompositeTransform::init();
    }

    QByteArray likely(const QByteArray &indentation) const
    {
        QByteArray result;
        result.append("{\n");
        foreach (Transform *t, transforms) {
            const QByteArray dst = t->likely(indentation + "  ");
            if (dst == "src")
                continue; // Not implemented
            result.append(indentation + "  src := ");
            result.append(dst);
            result.append("\n");
        }

        result.append(indentation + "  src\n}");
        return result;
    }

protected:
    // Template list project -- process templates in parallel through Transform::project
    // or if parallelism is disabled, handle them sequentially
   void _project(const TemplateList &src, TemplateList &dst) const
    {
        TemplateList ftes;
        dst = src;
        foreach (const Transform *f, transforms) {
            TemplateList res;
            f->project(dst, res);
            splitFTEs(res, ftes);
            dst = res;
        }
        dst.append(ftes);
    }

   // Single template const project, pass the template through each sub-transform, one after the other
   virtual void _project(const Template &src, Template &dst) const
   {
       dst = src;
       foreach (const Transform *f, transforms) {
           try {
               dst >> *f;
               if (dst.file.fte)
                   break;
           } catch (...) {
               qWarning("Exception triggered when processing %s with transform %s", qPrintable(src.file.flat()), qPrintable(f->objectName()));
               dst = Template(src.file);
               dst.file.fte = true;
           }
       }
   }
};

BR_REGISTER(Transform, PipeTransform)

} // namespace br

#include "core/pipe.moc"
