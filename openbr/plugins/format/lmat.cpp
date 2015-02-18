#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup formats
 * \brief Likely matrix format
 *
 * www.liblikely.org
 * \author Josh Klontz \cite jklontz
 */
class lmatFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        const likely_const_mat m = likely_read(qPrintable(file.name), likely_file_guess);
        const Template result(likelyToOpenCVMat(m));
        likely_release_mat(m);
        return result;
    }

    void write(const Template &t) const
    {
        const likely_const_mat m = likelyFromOpenCVMat(t);
        likely_write(m, qPrintable(file.name));
        likely_release_mat(m);
    }
};

BR_REGISTER(Format, lmatFormat)

} // namespace br

#include "format/lmat.moc"
