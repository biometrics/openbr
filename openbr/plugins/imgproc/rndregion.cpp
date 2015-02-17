#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Selects a random region.
 * \author Josh Klontz \cite jklontz
 */
class RndRegionTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        RNG &rng = theRNG();
        float size = rng.uniform(0.2f, 1.f);
        float x = rng.uniform(0.f, 1.f-size);
        float y = rng.uniform(0.f, 1.f-size);

        dst = src.m()(Rect(src.m().cols * x,
                           src.m().rows * y,
                           src.m().cols * size,
                           src.m().rows * size));
    }
};

BR_REGISTER(Transform, RndRegionTransform)

} // namespace br

#include "imgproc/rndregion.moc"
