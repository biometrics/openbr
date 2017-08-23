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

static void _projectList(const Transform *transform, const TemplateList *src, TemplateList *dst)
{
    transform->project(*src, *dst);
}

/*!
 * \brief DOCUMENT ME CHARLES
 * \author Unknown \cite unknown
 */
class DistributeTemplateTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform)
    BR_PROPERTY(br::Transform*, transform, NULL)

public:

    Transform *smartCopy(bool &newTransform)
    {
        if (!transform->timeVarying()) {
            newTransform = false;
            return this;
        }
        newTransform = true;

        DistributeTemplateTransform *output = new DistributeTemplateTransform;
        bool newChild = false;
        output->transform = transform->smartCopy(newChild);
        if (newChild)
            output->transform->setParent(output);

        return output;
    }

    void train(const QList<TemplateList> &data)
    {
        if (!transform->trainable) {
            qWarning("Attempted to train untrainable transform, nothing will happen.");
            return;
        }

        QList<TemplateList> separated;
        foreach (const TemplateList &list, data) {
            foreach (const Template &t, list) {
                separated.append(TemplateList());
                separated.last().append(t);
            }
        }

        transform->train(separated);
    }

    void project(const Template &src, Template &dst) const
    {
        TemplateList input;
        input.append(src);
        TemplateList output;
        project(input, output);

        if (output.size() != 1) qFatal("output contains more than 1 template");
        else dst = output[0];
    }

    // For each input template, form a single element TemplateList, push all those
    // lists through transform, and form dst by concatenating the results.
    // Process the single elemnt templates in parallel if parallelism is enabled.
    void project(const TemplateList &src, TemplateList &dst) const
    {
        // Pre-allocate output for each template
        QList<TemplateList> output_buffer;
        output_buffer.reserve(src.size());

        // Can't declare this local to the loop because it would go out of scope
        QList<TemplateList> input_buffer;
        input_buffer.reserve(src.size());

        QFutureSynchronizer<void> futures;

        for (int i =0; i < src.size();i++) {
            input_buffer.append(TemplateList());
            output_buffer.append(TemplateList());
        }
        QList<QFuture<void> > temp;
        temp.reserve(src.size());
        for (int i=0; i<src.size(); i++) {
            input_buffer[i].append(src[i]);

            if ((Globals->parallelism > 1) && (src.size() > 1)) temp.append(QtConcurrent::run(_projectList, transform, &input_buffer[i], &output_buffer[i]));
            else _projectList(transform, &input_buffer[i], &output_buffer[i]);
        }
        // We add the futures in reverse order, since in Qt 5.1 at least the
        // waiting thread will wait on them in the order added (which for uniform priority
        // threads is the order of execution), and we want the waiting thread to go in the opposite order
        // so that it can steal runnables and do something besides wait.
        for (int i = temp.size() - 1; i >= 0; i--) {
            futures.addFuture(temp[i]);
        }

        futures.waitForFinished();

        for (int i=0; i<src.size(); i++) dst.append(output_buffer[i]);
    }

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        this->project(src, dst);
        return;
    }

    void init()
    {
        if (!transform)
            return;

        trainable = transform->trainable;
    }

};
BR_REGISTER(Transform, DistributeTemplateTransform)

} // namespace br

#include "core/distributetemplate.moc"
