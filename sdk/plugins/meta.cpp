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

#include <QtConcurrentRun>
#include <openbr_plugin.h>

#include "core/common.h"
#include "core/opencvutils.h"
#include "core/qtutils.h"

using namespace cv;

namespace br
{

static TemplateList Expanded(const TemplateList &templates)
{
    TemplateList expanded;
    foreach (const Template &t, templates) {
        if (t.isEmpty()) {
            if (!t.file.getBool("enrollAll"))
                expanded.append(t);
            continue;
        }

        const bool fte = t.file.getBool("FTE");
        QList<QPointF> landmarks = t.file.landmarks();
        QList<QRectF> ROIs = t.file.ROIs();
        if (landmarks.size() % t.size() != 0) qFatal("Uneven landmark count.");
        if (ROIs.size() % t.size() != 0) qFatal("Uneven ROI count.");
        const int landmarkStep = landmarks.size() / t.size();
        const int ROIStep = ROIs.size() / t.size();

        for (int i=0; i<t.size(); i++) {
            if (!fte || !t.file.getBool("enrollAll")) {
                expanded.append(Template(t.file, t[i]));
                expanded.last().file.setROIs(ROIs.mid(i*ROIStep, ROIStep));
                expanded.last().file.setLandmarks(landmarks.mid(i*landmarkStep, landmarkStep));
            }
        }
    }
    return expanded;
}

static void _train(Transform *transform, const TemplateList *data)
{
    transform->train(*data);
}

// For handling progress feedback
static int depth = 0;

static void acquireStep()
{
    if (depth == 0) {
        Globals->currentStep = 0;
        Globals->totalSteps = 1;
    }
    depth++;
}

static void releaseStep()
{
    depth--;
    Globals->currentStep = floor(Globals->currentStep * pow(10.0, double(depth))) / pow(10.0, double(depth));
    if (depth == 0)
        Globals->totalSteps = 0;
}

static void incrementStep()
{
    Globals->currentStep += 1.0 / pow(10.0, double(depth));
}

/*!
 * \brief Use Expanded after basic calls that take a template list, used to implement ExpandTransform
 */
class ExpandDecorator : public Transform
{
    Q_OBJECT

    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform)
    BR_PROPERTY(br::Transform*, transform, NULL)

public:
    ExpandDecorator(Transform * input)
    {
        transform = input;
        transform->setParent(this);
        file = transform->file;
        setObjectName(transform->objectName());
    }

    void train(const TemplateList &data)
    {
        transform->train(data);
    }

    void project(const Template &src, Template &dst) const
    {
        transform->project(src, dst);
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        transform->project(src, dst);
        dst = Expanded(dst);
    }


    void projectUpdate(const Template &src, Template &dst)
    {
        transform->projectUpdate(src, dst);
    }

    void projectUpdate(const TemplateList & src, TemplateList & dst)
    {
        transform->projectUpdate(src, dst);
        dst = Expanded(dst);
    }

    bool timeVarying() const
    {
        return transform->timeVarying();
    }

    void finalize(TemplateList & output)
    {
        transform->finalize(output);
        output = Expanded(output);
    }

};

/*!
 * \brief A MetaTransform that aggregates some sub-transforms
 */
class BR_EXPORT CompositeTransform : public TimeVaryingTransform
{
    Q_OBJECT

public:
    Q_PROPERTY(QList<br::Transform*> transforms READ get_transforms WRITE set_transforms RESET reset_transforms)
    BR_PROPERTY(QList<br::Transform*>, transforms, QList<br::Transform*>())

    virtual void project(const Template &src, Template &dst) const
    {
        if (timeVarying()) qFatal("No const project defined for time-varying transform");
        _project(src, dst);
    }

    virtual void project(const TemplateList &src, TemplateList &dst) const
    {
        if (timeVarying()) qFatal("No const project defined for time-varying transform");
        _project(src, dst);
    }

    bool timeVarying() const { return isTimeVarying; }

    void init()
    {
        isTimeVarying = false;
        foreach (const br::Transform *transform, transforms) {
            if (transform->timeVarying()) {
                isTimeVarying = true;
                break;
            }
        }
    }

protected:
    bool isTimeVarying;

    virtual void _project(const Template & src, Template & dst) const = 0;
    virtual void _project(const TemplateList & src, TemplateList & dst) const = 0;

    CompositeTransform() : TimeVaryingTransform(false) {}
};

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

    void train(const TemplateList &data)
    {
        acquireStep();

        TemplateList copy(data);
        for (int i=0; i<transforms.size(); i++) {
            transforms[i]->train(copy);
            copy >> *transforms[i];
            incrementStep();
        }

        releaseStep();
    }

    void backProject(const Template &dst, Template &src) const
    {
        // Backprojecting a time-varying transform is probably not going to work.
        if (timeVarying()) qFatal("No backProject defined for time-varying transform");

        src = dst;
        // Reverse order in which transforms are processed
        int length = transforms.length();
        for (int i=length-1; i>=0; i--) {
            Transform *f = transforms.at(i);
            try {
                src >> *f;
            } catch (...) {
                qWarning("Exception triggered when processing %s with transform %s", qPrintable(dst.file.flat()), qPrintable(f->objectName()));
                src = Template(src.file);
                src.file.setBool("FTE");
            }
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
                dst.file.setBool("FTE");
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
        if (Globals->parallelism < 0) {
            dst = src;
            foreach (const Transform *f, transforms)
                dst >> *f;
        } else {
            Transform::project(src, dst);
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
               dst.file.setBool("FTE");
           }
       }
   }
};

BR_REGISTER(Transform, PipeTransform)

/*!
 * \ingroup transforms
 * \brief Transforms in series with expansion step.
 * \author Josh Klontz \cite jklontz
 *
 * The source br::Template is given to the first transform and the resulting br::Template is passed to the next transform, etc.
 * Each matrix is expanded into its own template between steps.
 *
 * \see PipeTransform
 */
class ExpandTransform : public PipeTransform
{
    Q_OBJECT

    void init()
    {
        for (int i = 0; i < transforms.size(); i++)
        {
            transforms[i] = new ExpandDecorator(transforms[i]);
        }
        // Need to call this to set up timevariance correctly, and it won't
        // be called automatically
        CompositeTransform::init();
    }

protected:

    // Template list project -- project through transforms sequentially,
    // then expand the results, can't use Transform::Project(templateList) since
    // we need to expand between tranforms, so actually do need to overload this method
    void _project(const TemplateList &src, TemplateList &dst) const
    {
        dst = src;
        for (int i=0; i<transforms.size(); i++) {
            dst >> *transforms[i];
        }
    }
};

BR_REGISTER(Transform, ExpandTransform)

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

    void train(const TemplateList &data)
    {
        QList< QFuture<void> > futures;
        const bool threaded = Globals->parallelism && (transforms.size() > 1);
        for (int i=0; i<transforms.size(); i++) {
            if (threaded) futures.append(QtConcurrent::run(_train, transforms[i], &data));
            else                                           _train (transforms[i], &data);
        }
        if (threaded) Globals->trackFutures(futures);
    }

    void backProject(const Template &dst, Template &src) const {Transform::backProject(dst, src);}

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
                dst.file.setBool("FTE");
            }
        }
    }

    void projectUpdate(const TemplateList & src, TemplateList & dst)
    {
        dst = src;
        dst.reserve(src.size());
        for (int i=0; i<src.size(); i++) dst.append(Template());
        foreach (Transform *f, transforms) {
            TemplateList m;
            f->projectUpdate(src, m);
            if (m.size() != dst.size()) qFatal("TemplateList is of an unexpected size.");
            for (int i=0; i<src.size(); i++) dst[i].append(m[i]);
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
                dst.file.setBool("FTE");
            }
        }
    }

    void _project(const TemplateList &src, TemplateList &dst) const
    {
        if (Globals->parallelism < 0) {
            dst.reserve(src.size());
            for (int i=0; i<src.size(); i++) dst.append(Template());
            foreach (const Transform *f, transforms) {
                TemplateList m;
                f->project(src, m);
                if (m.size() != dst.size()) qFatal("TemplateList is of an unexpected size.");
                for (int i=0; i<src.size(); i++) dst[i].append(m[i]);
            }
        } else {
            Transform::project(src, dst);
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

    void train(const TemplateList &data)
    {
        transform->train(data);
    }

    void project(const Template &src, Template &dst) const
    {
        const QString &file = src.file;
        if (cache.contains(file)) {
            dst = cache[file];
            dst.file.setLabel(src.file.label());
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
            vals.append(t.file.getFloat(transform->objectName()));
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
        const float val = projectedSrc.file.getFloat(transform->objectName());

        dst = src;
        dst.file.insert(transform->objectName(), val);
        dst.file.insert("FTE", (val < min) || (val > max));
    }
};

BR_REGISTER(Transform, FTETransform)

} // namespace br

#include "meta.moc"
