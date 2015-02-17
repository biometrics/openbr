#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup initializers
 * \brief Register custom objects with Qt meta object system.
 * \author Charles Otto \cite caotto
 */
class Registrar : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        qRegisterMetaType<br::Neighbors>();
    }
};

BR_REGISTER(Initializer, Registrar)

} // namespace br

#include "core/registrar.moc"
