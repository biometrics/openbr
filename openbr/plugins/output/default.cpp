#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup outputs
 * \brief Adaptor class -- write a matrix output using Format classes.
 * \author Charles Otto \cite caotto
 */
class DefaultOutput : public MatrixOutput
{
    Q_OBJECT

    ~DefaultOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;

        br::Template T(this->file, this->data);
        QScopedPointer<Format> writer(Factory<Format>::make(this->file));
        writer->write(T);
    }
};

BR_REGISTER(Output, DefaultOutput)

} // namespace br

#include "output/default.moc"
