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
#include <qbuffer.h>

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
        const bool enrollAll = t.file.get<bool>("enrollAll");
        if (t.isEmpty()) {
            if (!enrollAll)
                expanded.append(t);
            continue;
        }

        const QList<QPointF> points = t.file.points();
        const QList<QRectF> rects = t.file.rects();
        if (points.size() % t.size() != 0) qFatal("Uneven point count.");
        if (rects.size() % t.size() != 0) qFatal("Uneven rect count.");
        const int pointStep = points.size() / t.size();
        const int rectStep = rects.size() / t.size();

        for (int i=0; i<t.size(); i++) {
            expanded.append(Template(t.file, t[i]));
            expanded.last().file.setRects(rects.mid(i*rectStep, rectStep));
            expanded.last().file.setPoints(points.mid(i*pointStep, pointStep));
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
                qDebug() << "Training " << transforms[i]->description() << "\n...";
                transforms[i]->train(dataLines);
            }

            // if the transform is time varying, we can't project it in parallel
            if (transforms[i]->timeVarying()) {
                qDebug() << "Projecting " << transforms[i]->description() << "\n...";
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

    virtual void project(const Template &src, Template &dst) const
    {
        dst = src;
        qDebug("Called Expand project(Template,Template), nothing will happen");
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

        foreach (const Template &t, src) {
            out.merge(t);
        }
        out.file.clearRects();
        foreach (const Template &t, src) {
            if (!t.file.rects().empty())
                out.file.appendRects(t.file.rects());
        }
        dst.clear();
        dst.append(out);
    }

    virtual void project(const Template &src, Template &dst) const
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
    Q_PROPERTY(QString transformString READ get_transformString WRITE set_transformString RESET reset_transformString STORED false)
    Q_PROPERTY(QString fileName READ get_fileName WRITE set_fileName RESET reset_fileName STORED false)
    BR_PROPERTY(QString, transformString, "Identity")
    BR_PROPERTY(QString, fileName, QString())

public:
    Transform *transform;

    LoadStoreTransform() : transform(NULL) {}

    QString description(bool expanded = false) const
    {
        if (expanded) {
            QString res = transform->description(expanded);
            return res;
        }
        return br::Object::description(expanded);
    }

    Transform *simplify(bool &newTForm)
    {
        Transform *res = transform->simplify(newTForm);
        return res;
    }

    QList<Object *> getChildren() const
    {
        QList<Object *> rval;
        rval.append(transform);
        return rval;
    }
private:

    void init()
    {
        if (transform != NULL) return;
        if (fileName.isEmpty()) fileName = QRegExp("^[_a-zA-Z0-9]+$").exactMatch(transformString) ? transformString : QtUtils::shortTextHash(transformString);

        if (!tryLoad())
            transform = make(transformString);
        else
            trainable = false;
    }

    bool timeVarying() const
    {
        return transform->timeVarying();
    }

    void train(const QList<TemplateList> &data)
    {
        if (QFileInfo(getFileName()).exists())
            return;

        transform->train(data);

        qDebug("Storing %s", qPrintable(fileName));
        QtUtils::BlockCompression compressedOut;
        QFile fout(fileName);
        QtUtils::touchDir(fout);
        compressedOut.setBasis(&fout);

        QDataStream stream(&compressedOut);
        QString desc = transform->description();

        if (!compressedOut.open(QFile::WriteOnly))
            qFatal("Failed to open %s for writing.", qPrintable(file));

        stream << desc;
        transform->store(stream);
        compressedOut.close();
    }

    void project(const Template &src, Template &dst) const
    {
        transform->project(src, dst);
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        transform->project(src, dst);
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        transform->projectUpdate(src, dst);
    }

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        transform->projectUpdate(src, dst);
    }

    void finalize(TemplateList &output)
    {
        transform->finalize(output);
    }

    QString getFileName() const
    {
        if (QFileInfo(fileName).exists()) return fileName;
        const QString file = Globals->sdkPath + "/share/openbr/models/transforms/" + fileName;
        return QFileInfo(file).exists() ? file : QString();
    }

    bool tryLoad()
    {
        const QString file = getFileName();
        if (file.isEmpty()) return false;

        qDebug("Loading %s", qPrintable(file));
        QFile fin(file);
        QtUtils::BlockCompression reader(&fin);
        if (!reader.open(QIODevice::ReadOnly)) {
            if (QFileInfo(file).exists()) qFatal("Unable to open %s for reading. Check file permissions.", qPrintable(file));
            else            qFatal("Unable to open %s for reading. File does not exist.", qPrintable(file));
        }

        QDataStream stream(&reader);
        stream >> transformString;

        transform = Transform::make(transformString);
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
        dst.file.set("PossibleFTE", (val < min) || (val > max));
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

/*!
 * \ingroup transforms
 * \brief Generates two templates, one of which is passed through a transform and the other
 *        is not. No cats were harmed in the making of this transform.
 * \author Scott Klum \cite sklum
 */
class SchrodingerTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform)
    BR_PROPERTY(br::Transform*, transform, NULL)

public:
    void train(const TemplateList &data)
    {
        transform->train(data);
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach(const Template &t, src) {
            dst.append(t);
            Template u;
            transform->project(t,u);
            dst.append(u);
        }
    }

    void project(const Template &src, Template &dst) const {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

};
BR_REGISTER(Transform, SchrodingerTransform)

} // namespace br

#include "meta.moc"
