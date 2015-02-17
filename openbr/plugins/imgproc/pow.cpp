#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Raise each element to the specified power.
 * \author Josh Klontz \cite jklontz
 */
class PowTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float power READ get_power WRITE set_power RESET reset_power STORED false)
    Q_PROPERTY(bool preserveSign READ get_preserveSign WRITE set_preserveSign RESET reset_preserveSign STORED false)
    BR_PROPERTY(float, power, 2)
    BR_PROPERTY(bool, preserveSign, false)

    void project(const Template &src, Template &dst) const
    {
        pow(preserveSign ? abs(src) : src.m(), power, dst);
        if (preserveSign) subtract(Scalar::all(0), dst, dst, src.m() < 0);
    }
};

BR_REGISTER(Transform, PowTransform)

} // namespace br

#include "imgproc/pow.moc"
