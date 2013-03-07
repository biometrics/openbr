#include <openbr_plugin.h>

#include "core/common.h"

namespace br
{

/*!
 * \ingroup transforms
 * \brief Implements the YouTubesFaceDB \cite wolf11 experimental protocol.
 * \author Josh Klontz \cite jklontz
 */
class YouTubeFacesDBTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString algorithm READ get_algorithm WRITE set_algorithm RESET reset_algorithm STORED false)
    BR_PROPERTY(QString, algorithm, "")

    QSharedPointer<Transform> transform;
    QSharedPointer<Distance> distance;

    void init()
    {
        if (algorithm.isEmpty()) return;
        transform = Transform::fromAlgorithm(algorithm);
        distance = Distance::fromAlgorithm(algorithm);
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        Transform::project(src.mid(1) /* First template is the header in 'splits.txt' */, dst);
    }

    void project(const Template &src, Template &dst) const
    {
        const QStringList words = src.file.name.split(", ");
        dst.file.name = words[0] + "_" + words[1] + "_" + words[4] + ".mtx";

        TemplateList queryTemplates = TemplateList::fromGallery(File(words[2]).resolved());
        sort(queryTemplates);
        queryTemplates >> *transform;

        TemplateList targetTemplates = TemplateList::fromGallery(File(words[3]).resolved());
        sort(targetTemplates);
        targetTemplates >> *transform;

        QScopedPointer<MatrixOutput> memoryOutput(MatrixOutput::make(targetTemplates.files(), queryTemplates.files()));
        distance->compare(targetTemplates, queryTemplates, memoryOutput.data());

        dst.clear();
        dst.m() = memoryOutput.data()->data;
    }

    static void sort(TemplateList &templates)
    {
        // The file names in the YouTube Faces Database make it very difficult
        // for them to be ordered by frame number automatically,
        // hence we do it manually here.
        QList<int> frames;
        foreach (const Template &t, templates) {
            QStringList words = t.file.name.split('.');
            frames.append(words[words.size()-2].toInt());
        }

        typedef QPair<int,int> SortedFrame; // <frame number, original index>
        QList<SortedFrame> sortedFrames = Common::Sort(frames);
        TemplateList sortedTemplates; sortedTemplates.reserve(templates.size());
        foreach (const SortedFrame &sortedFrame, sortedFrames)
            sortedTemplates.append(templates[sortedFrame.second]);

        templates = sortedTemplates;
    }
};

BR_REGISTER(Transform, YouTubeFacesDBTransform)

} // namespace br

#include "youtube.moc"
