#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Concatenates all input matrices by row into a single matrix.
 * All matricies must have the same column counts.
 * \author Josh Klontz \cite jklontz
 */
class CatRowsTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = Template(src.file, OpenCVUtils::toMatByRow(src));
    }
};

BR_REGISTER(Transform, CatRowsTransform)

} // namespace br

#include "imgproc/catrows.moc"
