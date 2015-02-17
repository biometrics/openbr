#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Clears the points from a template
 * \author Brendan Klare \cite bklare
 */
class ClearPointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        dst.clearPoints();
    }
};

BR_REGISTER(Transform, ClearPointsTransform)

} // namespace br

#include "metadata/clearpoints.moc"
