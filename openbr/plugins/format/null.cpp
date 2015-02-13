#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup formats
 * \brief Returns an empty matrix.
 * \author Josh Klontz \cite jklontz
 */
class nullFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        return Template(file, cv::Mat());
    }

    void write(const Template &t) const
    {
        (void)t;
    }
};

BR_REGISTER(Format, nullFormat)

} // namespace br

#include "null.moc"
