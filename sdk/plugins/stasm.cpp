#include <openbr_plugin.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps STASM key point detector
 * \author Scott Klum \cite sklum
 */

class STASMTransform : public UntrainableTransform
{
    Q_OBJECT

    void init()
    {

    }

    void project(const Template &src, Template &dst) const
    {

    }
};

BR_REGISTER(Transform, STASMTransform)

} // namespace br

#include "stasm.moc"
