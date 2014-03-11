#include <likely.h>

#include "openbr_internal.h"

namespace br
{

class LikelyTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;
    }
};

BR_REGISTER(Transform, LikelyTransform)

}

#include "likely.moc"
