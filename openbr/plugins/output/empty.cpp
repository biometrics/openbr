#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup outputs
 * \brief Output to the terminal.
 * \author Josh Klontz \cite jklontz
 */
class EmptyOutput : public MatrixOutput
{
    Q_OBJECT

    static QString bufferString(const QString &string, int length)
    {
        if (string.size() >= length)
            return string.left(length);
        QString buffer; buffer.fill(' ', length-string.size());
        return string+buffer;
    }

    ~EmptyOutput()
    {
        if (targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        QString result;
        if ((queryFiles.size() == 1) && (targetFiles.size() == 1)) {
            result = toString(0,0) + "\n";
        } else {
            const int CELL_SIZE = 12;

            result = bufferString(" ", CELL_SIZE) + " ";
            foreach (const QString &targetName, targetFiles.names())
                result += bufferString(targetName, CELL_SIZE) + " ";
            result += "\n";

            for (int i=0; i<queryFiles.size(); i++) {
                result += bufferString(queryFiles[i].name, CELL_SIZE) + " ";
                for (int j=0; j<targetFiles.size(); j++)
                    result += bufferString(toString(i,j), CELL_SIZE) + " ";
                result += "\n";
            }
        }

        printf("%s", qPrintable(result));
    }
};

BR_REGISTER(Output, EmptyOutput)

} // namespace br

#include "output/empty.moc"
