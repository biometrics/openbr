#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Treats each line as a call to File::flat()
 * \author Josh Klontz \cite jklontz
 */
class flatGallery : public FileGallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        readOpen();
        *done = false;
        if (f.atEnd())
            f.seek(0);

        TemplateList templates;

        for (qint64 i = 0; i < readBlockSize; i++)
        {
            QByteArray line = f.readLine();

            if (!line.isEmpty()) {
                templates.append(File(QString::fromLocal8Bit(line).trimmed()));
                templates.last().file.set("progress", this->position());
            }

            if (f.atEnd()) {
                *done=true;
                break;
            }
        }

        return templates;
    }

    void write(const Template &t)
    {
        writeOpen();
        f.write((t.file.flat()+"\n").toLocal8Bit() );
    }
};

BR_REGISTER(Gallery, flatGallery)

} // namespace br

#include "gallery/flat.moc"
