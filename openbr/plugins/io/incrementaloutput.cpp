#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Incrementally output templates received to a gallery, based on the current filename
 * When a template is received in projectUpdate for the first time since a finalize, open a new gallery based on the
 * template's filename, and the galleryFormat property.
 * Templates received in projectUpdate will be output to the gallery with a filename combining their original filename and
 * their FrameNumber property, with the file extension specified by the fileFormat property.
 * \author Charles Otto \cite caotto
 */
class IncrementalOutputTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString galleryFormat READ get_galleryFormat WRITE set_galleryFormat RESET reset_galleryFormat STORED false)
    Q_PROPERTY(QString fileFormat READ get_fileFormat WRITE set_fileFormat RESET reset_fileFormat STORED false)
    BR_PROPERTY(QString, galleryFormat, "")
    BR_PROPERTY(QString, fileFormat, ".png")

    bool galleryUp;

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        if (src.empty())
            return;

        if (!galleryUp) {
            QFileInfo finfo(src[0].file.name);
            QString galleryName = finfo.baseName() + galleryFormat;

            writer = QSharedPointer<Gallery> (Factory<Gallery>::make(galleryName));
            galleryUp = true;
        }

        dst = src;
        int idx =0;
        foreach (const Template &t, src) {
            if (t.empty())
                continue;

            // Build the output filename for this template
            QFileInfo finfo(t.file.name);
            QString outputName = finfo.baseName() +"_" + t.file.get<QString>("FrameNumber") + "_" + QString::number(idx)+ fileFormat;

            idx++;
            Template out = t;
            out.file.name = outputName;
            writer->write(out);
        }
    }

    void train(const TemplateList& data)
    {
        (void) data;
    }

    // Drop the current gallery.
    void finalize(TemplateList &data)
    {
        (void) data;
        galleryUp = false;
    }

    QSharedPointer<Gallery> writer;
public:
    IncrementalOutputTransform() : TimeVaryingTransform(false,false) {galleryUp = false;}
};

BR_REGISTER(Transform, IncrementalOutputTransform)

} // namespace br

#include "io/incrementaloutput.moc"
