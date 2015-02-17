#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Combine all templates into one large matrix and process it as a br::Format
 * \author Josh Klontz \cite jklontz
 */
class matrixGallery : public Gallery
{
    Q_OBJECT
    Q_PROPERTY(const QString extension READ get_extension WRITE set_extension RESET reset_extension STORED false)
    BR_PROPERTY(QString, extension, "mtx")

    TemplateList templates;

    ~matrixGallery()
    {
        if (templates.isEmpty())
            return;

        QScopedPointer<Format> format(Factory<Format>::make(getFormat()));
        format->write(Template(file, OpenCVUtils::toMat(templates.data())));
    }

    File getFormat() const
    {
        return file.name.left(file.name.size() - file.suffix().size()) + extension;
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        return TemplateList() << getFormat();
    }

    void write(const Template &t)
    {
        templates.append(t);
    }
};

BR_REGISTER(Gallery, matrixGallery)

} // namespace br

#include "gallery/matrix.moc"
