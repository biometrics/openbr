#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Treats the gallery as a br::Format.
 * \author Josh Klontz \cite jklontz
 */
class DefaultGallery : public Gallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        *done = true;
        return TemplateList() << file;
    }

    void write(const Template &t)
    {
        QScopedPointer<Format> format(Factory<Format>::make(file));
        format->write(t);
    }

    qint64 totalSize()
    {
        return 1;
    }
};

BR_REGISTER(Gallery, DefaultGallery)

} //  namespace br

#include "gallery/default.moc"
