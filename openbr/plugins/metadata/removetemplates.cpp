#include <QRegularExpression>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Remove templates with the specified file extension or metadata value.
 * \author Josh Klontz \cite jklontz
 */
class RemoveTemplatesTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString regexp READ get_regexp WRITE set_regexp RESET reset_regexp STORED false)
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    BR_PROPERTY(QString, regexp, "")
    BR_PROPERTY(QString, key, "")

    void project(const Template &src, Template &dst) const
    {
        const QRegularExpression re(regexp);
        const QRegularExpressionMatch match = re.match(key.isEmpty() ? src.file.suffix() : src.file.get<QString>(key));
        if (match.hasMatch()) dst = Template();
        else                  dst = src;
    }
};

BR_REGISTER(Transform, RemoveTemplatesTransform)

} // namespace br

#include "metadata/removetemplates.moc"
