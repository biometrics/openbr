#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Reshape the each matrix to the specified number of rows.
 * \author Josh Klontz \cite jklontz
 */
class ReshapeTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int rows READ get_rows WRITE set_rows RESET reset_rows STORED false)
    BR_PROPERTY(int, rows, 1)

    void project(const Template &src, Template &dst) const
    {
        dst = src.m().reshape(src.m().channels(), rows);
    }
};

BR_REGISTER(Transform, ReshapeTransform)

} // namespace br;

#include "imgproc/reshape.moc"
