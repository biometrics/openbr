#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Treats each line as a file.
 * \author Josh Klontz \cite jklontz
 *
 * The entire line is treated as the file path. An optional label may be specified using a space ' ' separator:
 *
\verbatim
<FILE>
<FILE>
...
<FILE>
\endverbatim
 * or
\verbatim
<FILE> <LABEL>
<FILE> <LABEL>
...
<FILE> <LABEL>
\endverbatim
 * \see csvGallery
 */
class txtGallery : public FileGallery
{
    Q_OBJECT
    Q_PROPERTY(QString label READ get_label WRITE set_label RESET reset_label STORED false)
    BR_PROPERTY(QString, label, "")

    TemplateList readBlock(bool *done)
    {
        readOpen();
        *done = false;
        if (f.atEnd())
            f.seek(0);

        TemplateList templates;

        for (qint64 i = 0; i < readBlockSize; i++)
        {
            QByteArray lineBytes = f.readLine();
            QString line = QString::fromLocal8Bit(lineBytes).trimmed();

            if (!line.isEmpty()){
                int splitIndex = line.lastIndexOf(' ');
                if (splitIndex == -1) templates.append(File(line));
                else                  templates.append(File(line.mid(0, splitIndex), line.mid(splitIndex+1)));
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
        QString line = t.file.name;
        if (!label.isEmpty())
            line += " " + t.file.get<QString>(label);

        f.write((line+"\n").toLocal8Bit() );
    }
};

BR_REGISTER(Gallery, txtGallery)

} // namespace br

#include "gallery/txt.moc"
