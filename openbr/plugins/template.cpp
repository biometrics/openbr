#include "openbr_internal.h"
#include <QRegularExpression>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Retains only the values for the keys listed, to reduce template size
 * \author Scott Klum \cite sklum
 */
class RetainTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    BR_PROPERTY(QStringList, keys, QStringList())

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        foreach(const QString& localKey, dst.file.localKeys()) {
            if (!keys.contains(localKey)) dst.file.remove(localKey);
        }
    }
};

BR_REGISTER(Transform, RetainTransform)

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

#include "template.moc"
