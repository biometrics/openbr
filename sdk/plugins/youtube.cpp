#include <openbr_plugin.h>

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
        transform = Transform::fromAlgorithm(algorithm);
        distance = Distance::fromAlgorithm(algorithm);
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        // First input is the header in 'splits.txt'
        if (src.file.getInt("Input_Index") == 0) return;

        const QStringList words = src.file.name.split(", ");
        dst.file.name = words[0] + "_" + words[1] + "_" + words[4] + ".mtx";

        TemplateList queryTemplates = TemplateList::fromGallery(File(words[2]).resolved());
        queryTemplates >> *transform;

        TemplateList targetTemplates = TemplateList::fromGallery(File(words[3]).resolved());
        targetTemplates >> *transform;

        QScopedPointer<MatrixOutput> memoryOutput(MatrixOutput::make(targetTemplates.files(), queryTemplates.files()));
        distance->compare(targetTemplates, queryTemplates, memoryOutput.data());

        dst.clear();
        dst.m() = memoryOutput.data()->data;
    }
};

BR_REGISTER(Transform, YouTubeFacesDBTransform)

} // namespace br

#include "youtube.moc"
