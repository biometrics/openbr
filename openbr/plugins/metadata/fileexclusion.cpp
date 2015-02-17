#include <openbr/plugins/openbr_internal.h>

namespace br
{

class FileExclusionTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(QString exclusionGallery READ get_exclusionGallery WRITE set_exclusionGallery RESET reset_exclusionGallery STORED false)
    BR_PROPERTY(QString, exclusionGallery, "")

    QSet<QString> excluded;

    void project(const Template &, Template &) const
    {
        qFatal("FileExclusion can't do anything here");
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach (const Template &srcTemp, src) {
            if (!excluded.contains(srcTemp.file))
                dst.append(srcTemp);
        }
    }

    void init()
    {
        if (exclusionGallery.isEmpty())
            return;
        File rFile(exclusionGallery);
        rFile.remove("append");

        FileList temp = FileList::fromGallery(rFile);
        excluded = QSet<QString>::fromList(temp.names());
    }
};

BR_REGISTER(Transform, FileExclusionTransform)

} // namespace br

#include "metadata/fileexclusion.moc"
