#include <QRegularExpression>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Apply the input regular expression to the value of inputProperty, store the matched portion in outputProperty.
 * \author Charles Otto \cite caotto
 */
class RegexPropertyTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString regexp READ get_regexp WRITE set_regexp RESET reset_regexp STORED false)
    Q_PROPERTY(QString inputProperty READ get_inputProperty WRITE set_inputProperty RESET reset_inputProperty STORED false)
    Q_PROPERTY(QString outputProperty READ get_outputProperty WRITE set_outputProperty RESET reset_outputProperty STORED false)
    BR_PROPERTY(QString, regexp, "(.*)")
    BR_PROPERTY(QString, inputProperty, "name")
    BR_PROPERTY(QString, outputProperty, "Label")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        QRegularExpression re(regexp);
        QRegularExpressionMatch match = re.match(dst.get<QString>(inputProperty));
        if (!match.hasMatch())
            qFatal("Unable to match regular expression \"%s\" to base name \"%s\"!", qPrintable(regexp), qPrintable(dst.get<QString>(inputProperty)));
        dst.set(outputProperty, match.captured(match.lastCapturedIndex()));
    }
};

BR_REGISTER(Transform, RegexPropertyTransform)

} // namespace br

#include "metadata/regexproperty.moc"
