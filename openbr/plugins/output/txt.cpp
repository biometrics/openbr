#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup outputs
 * \brief Text file output.
 * \author Josh Klontz \cite jklontz
 */
class txtOutput : public MatrixOutput
{
    Q_OBJECT

    ~txtOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        QStringList lines;
        foreach (const File &file, queryFiles)
            lines.append(file.name + " " + file.get<QString>("Label"));
        QtUtils::writeFile(file, lines);
    }
};

BR_REGISTER(Output, txtOutput)

} // namespace br

#include "output/txt.moc"
