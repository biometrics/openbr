#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief dst = a*src+b
 * \author Josh Klontz \cite jklontz
 */
class MAddTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(double a READ get_a WRITE set_a RESET reset_a STORED false)
    Q_PROPERTY(double b READ get_b WRITE set_b RESET reset_b STORED false)
    BR_PROPERTY(double, a, 1)
    BR_PROPERTY(double, b, 0)

    void project(const Template &src, Template &dst) const
    {
        src.m().convertTo(dst.m(), src.m().depth(), a, b);
    }
};

BR_REGISTER(Transform, MAddTransform)

} // namespace br

#include "madd.moc"
