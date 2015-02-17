#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/bee.h>

namespace br
{

/*!
 * \ingroup formats
 * \brief Reads a NIST BEE similarity matrix.
 * \author Josh Klontz \cite jklontz
 */
class mtxFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        QString target, query;
        Template result = BEE::readMatrix(file, &target, &query);
        result.file.set("Target", target);
        result.file.set("Query", query);
        return result;
    }

    void write(const Template &t) const
    {
        BEE::writeMatrix(t, file);
    }
};

BR_REGISTER(Format, mtxFormat)

/*!
 * \ingroup formats
 * \brief Reads a NIST BEE mask matrix.
 * \author Josh Klontz \cite jklontz
 */
class maskFormat : public mtxFormat
{
    Q_OBJECT
};

BR_REGISTER(Format, maskFormat)

} // namespace br

#include "format/mtx.moc"
