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

#include <openbr_plugin.h>

#include "core/common.h"
#include "core/qtutils.h"

using namespace br;

/**** ALGORITHM_CORE ****/
struct AlgorithmCore
{
    QSharedPointer<Transform> transform;
    QSharedPointer<Distance> distance;

    AlgorithmCore(const QString &name)
    {
        this->name = name;
        init(name);
    }

    bool isClassifier() const
    {
        return distance.isNull();
    }

    void train(const File &input, const QString &model)
    {
        TemplateList data(TemplateList::fromGallery(input));

        if (transform.isNull()) qFatal("Null transform.");
        qDebug("%d training files", data.size());

        QTime time; time.start();
        qDebug("Training Enrollment");
        transform->train(data);

        if (!distance.isNull()) {
            qDebug("Projecting Enrollment");
            data >> *transform;

            qDebug("Training Comparison");
            distance->train(data);
        }

        if (!model.isEmpty()) {
            qDebug("Storing %s", qPrintable(QFileInfo(model).fileName()));
            store(model);
        }

        qDebug("Training Time (sec): %d", time.elapsed()/1000);
    }

    void store(const QString &model) const
    {
        // Create stream
        QByteArray data;
        QDataStream out(&data, QFile::WriteOnly);

        // Serialize algorithm to stream
        out << name;
        transform->store(out);
        const bool hasComparer = !distance.isNull();
        out << hasComparer;
        if (hasComparer) distance->store(out);
        out << Globals->classes;

        // Compress and save to file
        QtUtils::writeFile(model, data, -1);
    }

    void load(const QString &model)
    {
        // Load from file and decompress
        QByteArray data;
        QtUtils::readFile(model, data, true);

        // Create stream
        QDataStream in(&data, QFile::ReadOnly);

        // Load algorithm
        in >> name; init(Globals->abbreviations.contains(name) ? Globals->abbreviations[name] : name);
        transform->load(in);
        bool hasDistance; in >> hasDistance;
        if (hasDistance) distance->load(in);
        in >> Globals->classes;
    }

    File getMemoryGallery(const File &file) const
    {
        return name + file.baseName() + file.hash() + ".mem";
    }

    FileList enroll(File input, File gallery = File())
    {
        FileList fileList;
        if (gallery.isNull()) gallery = getMemoryGallery(input);

        QScopedPointer<Gallery> g(Gallery::make(gallery));
        if (g.isNull()) return FileList();

        if (gallery.contains("read") || gallery.contains("cache")) {
            fileList = g->files();
        }
        if (!fileList.isEmpty() && gallery.contains("cache"))
            return fileList;

        const TemplateList i(TemplateList::fromGallery(input));
        if (i.isEmpty()) return fileList; // Nothing to enroll

        if (transform.isNull()) qFatal("Null transform.");
        const int blocks = Globals->blocks(i.size());
        Globals->currentStep = 0;
        Globals->totalSteps = i.size();
        Globals->startTime.start();

        const bool noDuplicates = gallery.contains("noDuplicates");
        QStringList fileNames = noDuplicates ? fileList.names() : QStringList();
        const int subBlockSize = 4*std::max(1, Globals->parallelism);
        const int numSubBlocks = ceil(1.0*Globals->blockSize/subBlockSize);
        int totalCount = 0, failureCount = 0;
        double totalBytes = 0;
        for (int block=0; block<blocks; block++) {
            for (int subBlock = 0; subBlock<numSubBlocks; subBlock++) {
                TemplateList data = i.mid(block*Globals->blockSize + subBlock*subBlockSize, subBlockSize);
                if (data.isEmpty()) break;
                if (noDuplicates)
                    for (int i=data.size()-1; i>=0; i--)
                        if (fileNames.contains(data[i].file.name))
                            data.removeAt(i);
                const int numFiles = data.size();

                if (Globals->backProject) {
                    TemplateList backProjectedData;
                    transform->backProject(data, backProjectedData);
                    data = backProjectedData;
                } else {
                    data >> *transform;
                }

                g->writeBlock(data);
                const FileList newFiles = data.files();
                fileList.append(newFiles);

                totalCount += newFiles.size();
                failureCount += newFiles.failures();
                totalBytes += data.bytes<double>();
                Globals->currentStep += numFiles;
                Globals->printStatus();
            }
        }

        const float speed = 1000 * Globals->totalSteps / Globals->startTime.elapsed() / std::max(1, abs(Globals->parallelism));
        if (!Globals->quiet && (Globals->totalSteps > 1))
            fprintf(stderr, "\rSPEED=%.1e  SIZE=%.4g  FAILURES=%d/%d  \n",
                    speed, totalBytes/totalCount, failureCount, totalCount);
        Globals->totalSteps = 0;

        return fileList;
    }

    void retrieveOrEnroll(const File &file, QScopedPointer<Gallery> &gallery, FileList &galleryFiles)
    {
        if ((file.suffix() == "gal") || (file.suffix() == "mem")) {
            // Retrieve it
            gallery.reset(Gallery::make(file));
            galleryFiles = gallery->files();
        } else {
            // Was it already enrolled in memory?
            gallery.reset(Gallery::make(getMemoryGallery(file)));
            galleryFiles = gallery->files();
            if (!galleryFiles.isEmpty()) return;

            // Enroll it
            enroll(file);
            gallery.reset(Gallery::make(getMemoryGallery(file)));
            galleryFiles = gallery->files();
        }
    }

    void compare(File targetGallery, File queryGallery, File output)
    {
        if (output.exists() && output.getBool("cache")) return;
        if (queryGallery == ".") queryGallery = targetGallery;

        QScopedPointer<Gallery> t, q;
        FileList targetFiles, queryFiles;
        retrieveOrEnroll(targetGallery, t, targetFiles);
        retrieveOrEnroll(queryGallery, q, queryFiles);

        QScopedPointer<Output> o(Output::make(output, targetFiles, queryFiles));

        if (distance.isNull()) qFatal("Null distance.");
        Globals->currentStep = 0;
        Globals->totalSteps = double(targetFiles.size()) * double(queryFiles.size());
        Globals->startTime.start();

        int queryBlock = -1;
        bool queryDone = false;
        while (!queryDone) {
            queryBlock++;
            TemplateList queries = q->readBlock(&queryDone);

            int targetBlock = -1;
            bool targetDone = false;
            while (!targetDone) {
                targetBlock++;
                TemplateList targets = t->readBlock(&targetDone);

                o->setBlock(queryBlock, targetBlock);
                distance->compare(targets, queries, o.data());

                Globals->currentStep += double(targets.size()) * double(queries.size());
                Globals->printStatus();
            }
        }

        const float speed = 1000 * Globals->totalSteps / Globals->startTime.elapsed() / std::max(1, abs(Globals->parallelism));
        if (!Globals->quiet && (Globals->totalSteps > 1)) fprintf(stderr, "\rSPEED=%.1e  \n", speed);
        Globals->totalSteps = 0;
    }

private:
    QString name;

    QString getFileName(const QString &description) const
    {
        const QString file = Globals->sdkPath + "/share/openbr/models/algorithms/" + description;
        return QFileInfo(file).exists() ? file : QString();
    }

    void init(QString description)
    {
        // Check if a trained binary already exists for this algorithm
        const QString file = getFileName(description);
        if (!file.isEmpty()) description = file;

        if (QFileInfo(description).exists()) {
            if (Globals->verbose) qDebug("Loading %s", qPrintable(QFileInfo(description).fileName()));
            load(description);
            return;
        }

        // Expand abbreviated algorithms to their full strings
        if (Globals->abbreviations.contains(description))
            return init(Globals->abbreviations[description]);

        QStringList words = QtUtils::parse(description, ':');
        if (words.size() > 2) qFatal("Invalid algorithm format.");

        transform = QSharedPointer<Transform>(Transform::make(words[0], NULL));
        if (words.size() > 1) distance = QSharedPointer<Distance>(Distance::make(words[1], NULL));
    }
};


class AlgorithmManager : public Initializer
{
    Q_OBJECT

public:
    static QHash<QString, QSharedPointer<AlgorithmCore> > algorithms;
    static QMutex algorithmsLock;

    void initialize() const {}

    void finalize() const
    {
        algorithms.clear();
    }

    static QSharedPointer<AlgorithmCore> getAlgorithm(const QString &algorithm)
    {
        if (algorithm.isEmpty()) qFatal("No default algorithm set.");

        if (!algorithms.contains(algorithm)) {
            // Some algorithms are recursive, so we need to construct them outside the lock.
            QSharedPointer<AlgorithmCore> algorithmCore(new AlgorithmCore(algorithm));

            algorithmsLock.lock();
            if (!algorithms.contains(algorithm))
                algorithms.insert(algorithm, algorithmCore);
            algorithmsLock.unlock();
        }

        return algorithms[algorithm];
    }
};

QHash<QString, QSharedPointer<AlgorithmCore> > AlgorithmManager::algorithms;
QMutex AlgorithmManager::algorithmsLock;

BR_REGISTER(Initializer, AlgorithmManager)

bool br::IsClassifier(const QString &algorithm)
{
    qDebug("Checking if %s is a classifier", qPrintable(algorithm));
    return AlgorithmManager::getAlgorithm(algorithm)->isClassifier();
}

void br::Train(const File &input, const File &model)
{
    qDebug("Training on %s%s", qPrintable(input.flat()),
                               model.isNull() ? "" : qPrintable(" to " + model.flat()));
    AlgorithmManager::getAlgorithm(model.getString("algorithm"))->train(input, model);
}

FileList br::Enroll(const File &input, const File &gallery)
{
    qDebug("Enrolling %s%s", qPrintable(input.flat()),
                             gallery.isNull() ? "" : qPrintable(" to " + gallery.flat()));
    return AlgorithmManager::getAlgorithm(gallery.getString("algorithm"))->enroll(input, gallery);
}

void br::Compare(const File &targetGallery, const File &queryGallery, const File &output)
{
    qDebug("Comparing %s and %s%s", qPrintable(targetGallery.flat()),
                                    qPrintable(queryGallery.flat()),
                                    output.isNull() ? "" : qPrintable(" to " + output.flat()));
    AlgorithmManager::getAlgorithm(output.getString("algorithm"))->compare(targetGallery, queryGallery, output);
}

void br::Convert(const File &src, const File &dst)
{
    qDebug("Converting %s to %s", qPrintable(src.flat()), qPrintable(dst.flat()));
    QScopedPointer<Format> before(Factory<Format>::make(src));
    QScopedPointer<Format> after(Factory<Format>::make(dst));
    after->write(before->read());
}

void br::Cat(const QStringList &inputGalleries, const QString &outputGallery)
{
    qDebug("Concatenating %d galleries to %s", inputGalleries.size(), qPrintable(outputGallery));
    foreach (const QString &inputGallery, inputGalleries)
        if (inputGallery == outputGallery)
            qFatal("outputGallery must not be in inputGalleries.");
    QScopedPointer<Gallery> og(Gallery::make(outputGallery));
    foreach (const QString &inputGallery, inputGalleries) {
        QScopedPointer<Gallery> ig(Gallery::make(inputGallery));
        bool done = false;
        while (!done) og->writeBlock(ig->readBlock(&done));
    }
}

QSharedPointer<br::Transform> br::Transform::fromAlgorithm(const QString &algorithm)
{
    return AlgorithmManager::getAlgorithm(algorithm)->transform;
}

QSharedPointer<br::Distance> br::Distance::fromAlgorithm(const QString &algorithm)
{
    return AlgorithmManager::getAlgorithm(algorithm)->distance;
}

#include "core.moc"
