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

#include <QFutureSynchronizer>
#include <QRegularExpression>
#include <QtConcurrentRun>
#include "openbr_internal.h"
#include "openbr/core/common.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/qtutils.h"
#include "openbr/core/resource.h"

using namespace cv;

namespace br
{

static TemplateList Expanded(const TemplateList &templates)
{
    TemplateList expanded;
    foreach (const Template &t, templates) {
        if (t.isEmpty()) {
            if (!t.file.get<bool>("enrollAll", false))
                expanded.append(t);
            continue;
        }

        const bool fte = t.file.get<bool>("FTE", false);
        QList<QPointF> points = t.file.points();
        QList<QRectF> rects = t.file.rects();
        if (points.size() % t.size() != 0) qFatal("Uneven point count.");
        if (rects.size() % t.size() != 0) qFatal("Uneven rect count.");
        const int pointStep = points.size() / t.size();
        const int rectStep = rects.size() / t.size();

        for (int i=0; i<t.size(); i++) {
            if (!fte || !t.file.get<bool>("enrollAll", false)) {
                expanded.append(Template(t.file, t[i]));
                expanded.last().file.setRects(rects.mid(i*rectStep, rectStep));
                expanded.last().file.setPoints(points.mid(i*pointStep, pointStep));
            }
        }
    }
    return expanded;
}

static void _train(Transform *transform, const QList<TemplateList> *data)
{
    transform->train(*data);
}

/*!
 * \ingroup Transforms
 * \brief Transforms in series.
 * \author Josh Klontz \cite jklontz
 *
 * The source br::Template is given to the first transform and the resulting br::Template is passed to the next transform, etc.
 *
 * \see ExpandTransform
 * \see ForkTransform
 */
class PipeTransform : public CompositeTransform
{
    Q_OBJECT    

    void _projectPartial(TemplateList *srcdst, int startIndex, int stopIndex)
    {
        for (int i=startIndex; i<stopIndex; i++)
            *srcdst >> *transforms[i];
    }

    void train(const QList<TemplateList> &data)
    {
        if (!trainable) return;

        QList<TemplateList> dataLines(data);

        int i = 0;
        while (i < transforms.size()) {
            fprintf(stderr, "\n%s", qPrintable(transforms[i]->objectName()));

            // Conditional statement covers likely case that first transform is untrainable
            if (transforms[i]->trainable) {
                fprintf(stderr, " training...");
                transforms[i]->train(dataLines);
            }

            // if the transform is time varying, we can't project it in parallel
            if (transforms[i]->timeVarying()) {
                fprintf(stderr, "\n%s projecting...", qPrintable(transforms[i]->objectName()));
                for (int j=0; j < dataLines.size();j++)
                    transforms[i]->projectUpdate(dataLines[j], dataLines[j]);

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

            fprintf(stderr, " projecting...");
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
            } catch (...) {
                qWarning("Exception triggered when processing %s with transform %s", qPrintable(src.file.flat()), qPrintable(f->objectName()));
                dst = Template(src.file);
                dst.file.set("FTE", true);
            }
        }
    }

    // For time varying transforms, parallel execution over individual templates
    // won't work.
    void projectUpdate(const TemplateList & src, TemplateList & dst)
    {
        dst = src;
        foreach (Transform *f, transforms)
        {
            f->projectUpdate(dst);
        }
    }

    virtual void finalize(TemplateList & output)
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

protected:
    // Template list project -- process templates in parallel through Transform::project
    // or if parallelism is disabled, handle them sequentially
   void _project(const TemplateList &src, TemplateList &dst) const
    {
        dst = src;
        foreach (const Transform *f, transforms) {
            dst >> *f;
        }
    }

   // Single template const project, pass the template through each sub-transform, one after the other
   virtual void _project(const Template & src, Template & dst) const
   {
       dst = src;
       foreach (const Transform *f, transforms) {
           try {
               dst >> *f;
           } catch (...) {
               qWarning("Exception triggered when processing %s with transform %s", qPrintable(src.file.flat()), qPrintable(f->objectName()));
               dst = Template(src.file);
               dst.file.set("FTE", true);
           }
       }
   }
};

BR_REGISTER(Transform, PipeTransform)

/*!
 * \ingroup transforms
 * \brief Performs an expansion step on input templatelists
 * \author Josh Klontz \cite jklontz
 *
 * Each matrix in an input Template is expanded into its own template.
 *
 * \see PipeTransform
 */
class ExpandTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    virtual void project(const TemplateList &src, TemplateList &dst) const
    {
        dst = Expanded(src);
    }

    virtual void project(const Template & src, Template & dst) const
    {
        qFatal("this has gone bad");
        (void) src; (void) dst;
    }
};

BR_REGISTER(Transform, ExpandTransform)

/*!
 * \ingroup transforms
 * \brief It's like the opposite of ExpandTransform, but not really
 * \author Charles Otto \cite caotto
 *
 * Given a set of templatelists as input, concatenate them onto a single Template
 */
class ContractTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    virtual void project(const TemplateList &src, TemplateList &dst) const
    {
        if (src.empty()) return;
        Template out;

        foreach (const Template & t, src) {
            out.merge(t);
        }
        dst.clear();
        dst.append(out);
    }

    virtual void project(const Template & src, Template & dst) const
    {
        qFatal("this has gone bad");
        (void) src; (void) dst;
    }
};

BR_REGISTER(Transform, ContractTransform)

/*!
 * \ingroup transforms
 * \brief Transforms in parallel.
 * \author Josh Klontz \cite jklontz
 *
 * The source br::Template is seperately given to each transform and the results are appended together.
 *
 * \see PipeTransform
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
    void projectupdate(const Template & src, Template & dst)
    {
        foreach (Transform *f, transforms) {
            try {
                Template res;
                f->projectUpdate(src, res);
                dst.merge(res);
            } catch (...) {
                qWarning("Exception triggered when processing %s with transform %s", qPrintable(src.file.flat()), qPrintable(f->objectName()));
                dst = Template(src.file);
                dst.file.set("FTE", true);
            }
        }
    }

    void projectUpdate(const TemplateList & src, TemplateList & dst)
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
    virtual void finalize(TemplateList & output)
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
                dst.file.set("FTE", true);
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

/*!
 * \ingroup transforms
 * \brief Caches br::Transform::project() results.
 * \author Josh Klontz \cite jklontz
 */
class CacheTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform)
    BR_PROPERTY(br::Transform*, transform, NULL)

    static QHash<QString, Template> cache;
    static QMutex cacheLock;

public:
    ~CacheTransform()
    {
        if (cache.isEmpty()) return;

        // Write to cache
        QFile file("Cache");
        if (!file.open(QFile::WriteOnly))
            qFatal("Unable to open %s for writing.", qPrintable(file.fileName()));
        QDataStream stream(&file);
        stream << cache;
        file.close();
    }

private:
    void init()
    {
        if (!transform) return;

        trainable = transform->trainable;
        if (!cache.isEmpty()) return;

        // Read from cache
        QFile file("Cache");
        if (file.exists()) {
            if (!file.open(QFile::ReadOnly))
                qFatal("Unable to open %s for reading.", qPrintable(file.fileName()));
            QDataStream stream(&file);
            stream >> cache;
            file.close();
        }
    }

    void train(const QList<TemplateList> &data)
    {
        transform->train(data);
    }

    void project(const Template &src, Template &dst) const
    {
        const QString &file = src.file;
        if (cache.contains(file)) {
            dst = cache[file];
        } else {
            transform->project(src, dst);
            cacheLock.lock();
            cache[file] = dst;
            cacheLock.unlock();
        }
    }
};

QHash<QString, Template> CacheTransform::cache;
QMutex CacheTransform::cacheLock;

BR_REGISTER(Transform, CacheTransform)

/*!
 * \ingroup transforms
 * \brief Caches transform training.
 * \author Josh Klontz \cite jklontz
 */
class LoadStoreTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description STORED false)
    BR_PROPERTY(QString, description, "Identity")

    Transform *transform;
    QString baseName;

public:
    LoadStoreTransform() : transform(NULL) {}

private:
    void init()
    {
        if (transform != NULL) return;
        baseName = QRegExp("^[a-zA-Z0-9]+$").exactMatch(description) ? description : QtUtils::shortTextHash(description);
        if (!tryLoad()) transform = make(description);
        else            trainable = false;
    }

    void train(const TemplateList &data)
    {
        if (QFileInfo(getFileName()).exists())
            return;

        transform->train(data);

        qDebug("Storing %s", qPrintable(baseName));
        QByteArray byteArray;
        QDataStream stream(&byteArray, QFile::WriteOnly);
        stream << description;
        transform->store(stream);
        QtUtils::writeFile(baseName, byteArray, -1);
    }

    void project(const Template &src, Template &dst) const
    {
        transform->project(src, dst);
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        transform->project(src, dst);
    }

    QString getFileName() const
    {
        if (QFileInfo(baseName).exists()) return baseName;
        const QString file = Globals->sdkPath + "/share/openbr/models/transforms/" + baseName;
        return QFileInfo(file).exists() ? file : QString();
    }

    bool tryLoad()
    {
        const QString file = getFileName();
        if (file.isEmpty()) return false;

        if (Globals->verbose) qDebug("Loading %s", qPrintable(baseName));
        QByteArray data;
        QtUtils::readFile(file, data, true);
        QDataStream stream(&data, QFile::ReadOnly);
        stream >> description;
        transform = Transform::make(description);
        transform->load(stream);
        return true;
    }
};

BR_REGISTER(Transform, LoadStoreTransform)

/*!
 * \ingroup transforms
 * \brief Flags images that failed to enroll based on the specified transform.
 * \author Josh Klontz \cite jklontz
 */
class FTETransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform)
    Q_PROPERTY(float min READ get_min WRITE set_min RESET reset_min)
    Q_PROPERTY(float max READ get_max WRITE set_max RESET reset_max)
    BR_PROPERTY(br::Transform*, transform, NULL)
    BR_PROPERTY(float, min, -std::numeric_limits<float>::max())
    BR_PROPERTY(float, max,  std::numeric_limits<float>::max())

    void train(const TemplateList &data)
    {
        transform->train(data);

        TemplateList projectedData;
        transform->project(data, projectedData);

        QList<float> vals;
        foreach (const Template &t, projectedData) {
            if (!t.file.contains(transform->objectName()))
                qFatal("Matrix metadata missing key %s.", qPrintable(transform->objectName()));
            vals.append(t.file.get<float>(transform->objectName()));
        }
        float q1, q3;
        Common::Median(vals, &q1, &q3);
        min = q1 - 1.5 * (q3 - q1);
        max = q3 + 1.5 * (q3 - q1);
    }

    void project(const Template &src, Template &dst) const
    {
        Template projectedSrc;
        transform->project(src, projectedSrc);
        const float val = projectedSrc.file.get<float>(transform->objectName());

        dst = src;
        dst.file.set(transform->objectName(), val);
        dst.file.set("FTE", (val < min) || (val > max));
    }
};

BR_REGISTER(Transform, FTETransform)


static void _projectList(const Transform *transform, const TemplateList *src, TemplateList *dst)
{
    transform->project(*src, *dst);
}

class DistributeTemplateTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform)
    BR_PROPERTY(br::Transform*, transform, NULL)

public:

    Transform * smartCopy()
    {
        if (!transform->timeVarying())
            return this;

        DistributeTemplateTransform * output = new DistributeTemplateTransform;
        output->transform = transform->smartCopy();
        return output;
    }

    void train(const QList<TemplateList> &data)
    {
        if (!transform->trainable) {
            qWarning("Attempted to train untrainable transform, nothing will happen.");
            return;
        }

        QList<TemplateList> separated;
        foreach (const TemplateList & list, data) {
            foreach(const Template & t, list) {
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

            if (Globals->parallelism > 1) temp.append(QtConcurrent::run(_projectList, transform, &input_buffer[i], &output_buffer[i]));
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

#include "meta.moc"
