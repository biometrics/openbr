#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Set the last matrix of the input template to a matrix stored as metadata with input propName.
 *
 * Also removes the property from the templates metadata after restoring it.
 *
 * \author Charles Otto \cite caotto
 */
class RestoreMatTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString propName READ get_propName WRITE set_propName RESET reset_propName STORED false)
    BR_PROPERTY(QString, propName, "")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        if (dst.file.contains(propName)) {
            dst.clear();
            dst.m() = dst.file.get<cv::Mat>(propName);
            dst.file.remove(propName);
        }
    }
};

BR_REGISTER(Transform, RestoreMatTransform)

} // namespace br

#include "metadata/restoremat.moc"
