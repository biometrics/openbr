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
using namespace br;

static TemplateList Simplified(const TemplateList &templates)
{
    TemplateList simplified;
    foreach (const Template &t, templates) {
        if (t.isEmpty()) {
            if (!t.file.getBool("enrollAll"))
                simplified.append(t);
            continue;
        }

        const bool fte = t.file.getBool("FTE");
        QList<QPointF> landmarks = t.file.landmarks();
        QList<QRectF> ROIs = t.file.ROIs();
        if (landmarks.size() % t.size() != 0) qFatal("TemplateList::simplified uneven landmark count.");
        if (ROIs.size() % t.size() != 0) qFatal("TemplateList::simplified uneven ROI count.");
        const int landmarkStep = landmarks.size() / t.size();
        const int ROIStep = ROIs.size() / t.size();

        for (int i=0; i<t.size(); i++) {
            if (!fte || !t.file.getBool("enrollAll")) {
                simplified.append(Template(t.file, t[i]));
                simplified.last().file.setROIs(ROIs.mid(i*ROIStep, ROIStep));
                simplified.last().file.setLandmarks(landmarks.mid(i*landmarkStep, landmarkStep));
            }
        }
    }
    return simplified;
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
 * \ingroup Transforms
 * \brief Transforms in series.
 * \author Josh Klontz \cite jklontz
 *
 * The source br::Template is given to the first transform and the resulting br::Template is passed to the next transform, etc.
 *
 * \see ChainTransform
 */
class PipeTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<br::Transform*> transforms READ get_transforms WRITE set_transforms RESET reset_transforms)
    BR_PROPERTY(QList<br::Transform*>, transforms, QList<br::Transform*>())

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

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        foreach (const Transform *f, transforms) {
            try {
                dst >> *f;
            } catch (...) {
                qWarning("Exception triggered when processing %s with transform %s", qPrintable(src.file.flat()), qPrintable(f->name()));
                dst = Template(src.file);
                dst.file.setBool("FTE");
            }
        }
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        if (Globals->parallelism < 0) {
            dst = src;
            foreach (const Transform *f, transforms)
                dst >> *f;
        } else {
            Transform::project(src, dst);
        }
    }
};

BR_REGISTER(Transform, PipeTransform)

/*!
 * \ingroup transforms
 * \brief Transforms in series.
 * \author Josh Klontz \cite jklontz
 *
 * The source br::Template is given to the first transform and the resulting br::Template is passed to the next transform, etc.
 * Each matrix is reorganized into a new template before continuing.
 *
 * \see PipeTransform
 */
class ChainTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<br::Transform*> transforms READ get_transforms WRITE set_transforms RESET reset_transforms)
    BR_PROPERTY(QList<br::Transform*>, transforms, QList<br::Transform*>())

    void train(const TemplateList &data)
    {
        acquireStep();

        TemplateList copy(data);
        for (int i=0; i<transforms.size(); i++) {
            transforms[i]->train(copy);
            copy >> *transforms[i];
            copy = Simplified(copy);
            incrementStep();
        }

        releaseStep();
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        foreach (const Transform *f, transforms) {
            try {
                dst >> *f;
            } catch (...) {
                qWarning("Exception triggered when processing %s with transform %s", qPrintable(src.file.flat()), qPrintable(f->name()));
                dst = Template(src.file);
                dst.file.setBool("FTE");
            }
        }
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        dst = src;
        for (int i=0; i<transforms.size(); i++) {
            dst >> *transforms[i];
            dst = Simplified(dst);
        }
    }
};

BR_REGISTER(Transform, ChainTransform)

/*!
 * \ingroup transforms
 * \brief Transforms in parallel.
 * \author Josh Klontz \cite jklontz
 *
 * The source br::Template is seperately given to each transform and the results are appended together.
 */
class ForkTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<br::Transform*> transforms READ get_transforms WRITE set_transforms RESET reset_transforms)
    BR_PROPERTY(QList<br::Transform*>, transforms, QList<br::Transform*>())

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

    void project(const Template &src, Template &dst) const
    {
        foreach (const Transform *f, transforms) {
            try {
                dst.merge((*f)(src));
            } catch (...) {
                qWarning("Exception triggered when processing %s with transform %s", qPrintable(src.file.flat()), qPrintable(f->name()));
                dst = Template(src.file);
                dst.file.setBool("FTE");
            }
        }
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        if (Globals->parallelism < 0) {
            dst.reserve(src.size());
            for (int i=0; i<src.size(); i++) dst.append(Template());
            foreach (const Transform *f, transforms) {
                TemplateList m;
                f->project(src, m);
                if (m.size() != dst.size()) qFatal("Fork::project templateList is of an unexpected size.");
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
        bool success = file.open(QFile::WriteOnly); if (!success) qFatal("Cache::Cache unable to open %s for writing.", qPrintable(file.fileName()));
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
            bool success = file.open(QFile::ReadOnly); if (!success) qFatal("Cache::make unable to open %s for reading.", qPrintable(file.fileName()));
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
    Transform *transform = NULL;

    QString baseName;

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
        QtUtils::writeFile(baseName, byteArray);
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

        qDebug("Loading %s", qPrintable(baseName));
        QByteArray data;
        QtUtils::readFile(file, data);
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
            if (!t.file.contains(transform->name())) qFatal("FTE::train matrix metadata missing key %s.", qPrintable(transform->name()));
            vals.append(t.file.getFloat(transform->name()));
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
        const float val = projectedSrc.file.getFloat(transform->name());

        dst = src;
        dst.file.insert(transform->name(), val);
        dst.file.insert("FTE", (val < min) || (val > max));
    }
};

BR_REGISTER(Transform, FTETransform)

#include "meta.moc"
