#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Treat the file as a single binary template.
 * \author Josh Klontz \cite jklontz
 */
class templateGallery : public Gallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        *done = true;
        QByteArray data;
        QtUtils::readFile(file.name.left(file.name.size()-QString(".template").size()), data);
        return TemplateList() << Template(file, cv::Mat(1, data.size(), CV_8UC1, data.data()).clone());
    }

    void write(const Template &t)
    {
        (void) t;
        qFatal("Not supported.");
    }

    void init()
    {
        //
    }
};

BR_REGISTER(Gallery, templateGallery)

} // namespace br

#include "gallery/template.moc"
