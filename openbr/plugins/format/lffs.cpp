#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup formats
 * \brief Reads a NIST LFFS file.
 * \author Josh Klontz \cite jklontz
 */
class lffsFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        QByteArray byteArray;
        QtUtils::readFile(file.name, byteArray);
        return Mat(1, byteArray.size(), CV_8UC1, byteArray.data()).clone();
    }

    void write(const Template &t) const
    {
        QByteArray byteArray((const char*)t.m().data, t.m().total()*t.m().elemSize());
        QtUtils::writeFile(file.name, byteArray);
    }
};

BR_REGISTER(Format, lffsFormat)

} // namespace br

#include "format/lffs.moc"
