#include "openbr_internal.h"
#include <QRegularExpression>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Retains only the values for the keys listed, to reduce template size
 * \author Scott Klum \cite sklum
 */
class KeepMetadataTransform : public UntrainableTransform
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

BR_REGISTER(Transform, KeepMetadataTransform)

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
        const QRegularExpressionMatch match = re.match(key.isEmpty() ? src.file.baseName() : src.file.get<QString>(key));
        if (!match.hasMatch()) dst = Template();
        else                  dst = src;
    }
};

BR_REGISTER(Transform, RemoveTemplatesTransform)

/*!
 * \ingroup transforms
 * \brief Removes a metadata field from all templates
 * \author Brendan Klare \cite bklare
 */
class RemoveMetadataTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString attributeName READ get_attributeName WRITE set_attributeName RESET reset_attributeName STORED false)
    BR_PROPERTY(QString, attributeName, "None")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        if (dst.file.contains(attributeName))
            dst.file.remove(attributeName);
    }
};
BR_REGISTER(Transform, RemoveMetadataTransform)

/*!
 * \ingroup transforms
 * \brief Retains only landmarks/points at the provided indices
 * \author Brendan Klare \cite bklare
 */
class SelectPointsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    BR_PROPERTY(QList<int>, indices, QList<int>())

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        QList<QPointF> origPoints = src.file.points();
        dst.file.clearPoints();
        for (int i = 0; i < indices.size(); i++)
            dst.file.appendPoint(origPoints[indices[i]]);
    }
};

BR_REGISTER(Transform, SelectPointsTransform)

/*!
 * \ingroup transforms
 * \brief Converts Amazon MTurk labels
 * \author Scott Klum \cite sklum
 */
class MTurkTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(float maxVotes READ get_maxVotes WRITE set_maxVotes RESET reset_maxVotes STORED false)
    Q_PROPERTY(bool classify READ get_classify WRITE set_classify RESET reset_classify STORED false)
    Q_PROPERTY(bool consensusOnly READ get_consensusOnly WRITE set_consensusOnly RESET reset_consensusOnly STORED false)
    BR_PROPERTY(QString, inputVariable, QString())
    BR_PROPERTY(float, maxVotes, 1.)
    BR_PROPERTY(bool, classify, false)
    BR_PROPERTY(bool, consensusOnly, false)

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        QMap<QString,QVariant> map = dst.file.get<QMap<QString,QVariant> >(inputVariable);

        bool ok;

        QMapIterator<QString, QVariant> i(map);
        while (i.hasNext()) {
            i.next();
            // Normalize to [-1,1]
            float value = i.value().toFloat(&ok)/maxVotes;//* 2./maxVotes - 1;
            if (classify) (value > 0) ? value = 1 : value = -1;
            else if (consensusOnly && (value != 1 && value != -1)) continue;
            dst.file.set(i.key(),value);
        }
    }
};

BR_REGISTER(Transform, MTurkTransform)

} // namespace br

#include "template.moc"
