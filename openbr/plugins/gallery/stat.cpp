#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Print template statistics.
 * \author Josh Klontz \cite jklontz
 */
class statGallery : public Gallery
{
    Q_OBJECT
    QSet<QString> subjects;
    QList<int> bytes;

    ~statGallery()
    {
        int emptyTemplates = 0;
        for (int i=bytes.size()-1; i>=0; i--)
            if (bytes[i] == 0) {
                bytes.removeAt(i);
                emptyTemplates++;
            }

        double bytesMean, bytesStdDev;
        Common::MeanStdDev(bytes, &bytesMean, &bytesStdDev);
        printf("Subjects: %d\nEmpty Templates: %d/%d\nBytes/Template: %.4g +/- %.4g\n",
               subjects.size(), emptyTemplates, emptyTemplates+bytes.size(), bytesMean, bytesStdDev);
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        return TemplateList() << file;
    }

    void write(const Template &t)
    {
        subjects.insert(t.file.get<QString>("Label"));
        bytes.append(t.bytes());
    }
};

BR_REGISTER(Gallery, statGallery)

} // namespace br

#include "gallery/stat.moc"
