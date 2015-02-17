#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup outputs
 * \brief Discards the scores.
 * \author Josh Klontz \cite jklontz
 */
class nullOutput : public Output
{
    Q_OBJECT

    void set(float value, int i, int j)
    {
        (void) value; (void) i; (void) j;
    }
};

BR_REGISTER(Output, nullOutput)

} // namespace br

#include "output/null.moc"
