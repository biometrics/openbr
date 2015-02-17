#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Alpha-blend two matrices
 * \author Josh Klontz \cite jklontz
 */
class BlendTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(float alpha READ get_alpha WRITE set_alpha RESET reset_alpha STORED false)
    BR_PROPERTY(float, alpha, 0.5)

    void project(const Template &src, Template &dst) const
    {
        if (src.size() != 2) qFatal("Expected two source matrices.");
        addWeighted(src[0], alpha, src[1], 1-alpha, 0, dst);
    }
};

BR_REGISTER(Transform, BlendTransform)

} // namespace br

#include "imgproc/blend.moc"
