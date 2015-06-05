#include <openbr/plugins/openbr_internal.h>

#include <likely.h>
#include <likely/opencv.hpp>

namespace br
{

/*!
 * \ingroup formats
 * \brief Likely matrix format
 *
 * \br_link www.liblikely.org
 * \author Josh Klontz \cite jklontz
 */
class lmFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        const likely_const_mat m = likely_read(qPrintable(file.name), likely_file_guess, likely_void);
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

BR_REGISTER(Format, lmFormat)

} // namespace br

#include "format/lm.moc"
