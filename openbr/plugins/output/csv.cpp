#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup outputs
 * \brief Comma separated values output.
 * \author Josh Klontz \cite jklontz
 */
class csvOutput : public MatrixOutput
{
    Q_OBJECT

    ~csvOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        QStringList lines;
        lines.append("File," + targetFiles.names().join(","));
        for (int i=0; i<queryFiles.size(); i++) {
            QStringList words;
            for (int j=0; j<targetFiles.size(); j++)
                words.append(toString(i,j));  // The toString idiom is used to output match scores - see MatrixOutput
            lines.append(queryFiles[i].name+","+words.join(","));
        }
        QtUtils::writeFile(file, lines);
    }
};

BR_REGISTER(Output, csvOutput)

} // namespace br

#include "output/csv.moc"
