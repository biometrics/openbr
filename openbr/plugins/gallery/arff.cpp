#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Weka ARFF file format.
 * \author Josh Klontz \cite jklontz
 * http://weka.wikispaces.com/ARFF+%28stable+version%29
 */
class arffGallery : public Gallery
{
    Q_OBJECT
    QFile arffFile;

    TemplateList readBlock(bool *done)
    {
        (void) done;
        qFatal("Not implemented.");
        return TemplateList();
    }

    void write(const Template &t)
    {
        if (!arffFile.isOpen()) {
            arffFile.setFileName(file.name);
            arffFile.open(QFile::WriteOnly);
            arffFile.write("% OpenBR templates\n"
                           "@RELATION OpenBR\n"
                           "\n");

            const int dimensions = t.m().rows * t.m().cols;
            for (int i=0; i<dimensions; i++)
                arffFile.write(qPrintable("@ATTRIBUTE v" + QString::number(i) + " REAL\n"));
            arffFile.write(qPrintable("@ATTRIBUTE class string\n"));

            arffFile.write("\n@DATA\n");
        }

        arffFile.write(qPrintable(OpenCVUtils::matrixToStringList(t).join(',')));
        arffFile.write(qPrintable(",'" + t.file.get<QString>("Label") + "'\n"));
    }
};

BR_REGISTER(Gallery, arffGallery)

} // namespace br

#include "gallery/arff.moc"
