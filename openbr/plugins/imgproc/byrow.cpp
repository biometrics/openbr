#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Turns each row into its own matrix.
 * \author Josh Klontz \cite jklontz
 */
class ByRowTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        for (int i=0; i<src.m().rows; i++)
            dst += src.m().row(i);
    }
};

BR_REGISTER(Transform, ByRowTransform)

} // namespace br

#include "imgproc/byrow.moc"
