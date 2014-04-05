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
 * \brief Converts Amazon MTurk labels to a non-map format for use in a transform
 *        Also optionally normalizes and/or classifies the votes
 * \author Scott Klum \cite sklum
 */
/*
class TurkTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(QString outputVariable READ get_outputVariable WRITE set_outputVariable RESET reset_outputVariable STORED false)
    Q_PROPERTY(float maxVotes READ get_maxVotes WRITE set_maxVotes RESET reset_maxVotes STORED false)
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform STORED true)
    Q_PROPERTY(bool classify READ get_classify WRITE set_classify RESET reset_classify STORED false)
    Q_PROPERTY(bool consensusOnly READ get_consensusOnly WRITE set_consensusOnly RESET reset_consensusOnly STORED false)
    BR_PROPERTY(QString, inputVariable, QString())
    BR_PROPERTY(QString, outputVariable, QString())
    BR_PROPERTY(float, maxVotes, 1.)
    BR_PROPERTY(br::Transform*, transform, NULL)
    BR_PROPERTY(bool, classify, false)
    BR_PROPERTY(bool, consensusOnly, false)

    void train(const TemplateList &data)
    {
        TemplateList expandedData;

        foreach(const Template &t, data)
            expandedData.append(expandVotes(t));

        transform->train(expandedData);
    }

    void project(const Template &src, Template &dst) const
    {
        // Unmap, project, remap
        transform->project(expandVotes(src),dst);

        QMap<QString,QVariant> map = src.file.get<QMap<QString,QVariant> >(inputVariable);
        // We expect that whatever transform does to the inputVariable,
        // the outputVariable will be in the form of (or convertible to) a float
        map.insert(outputVariable,dst.file.get<float>(outputVariable));

        dst = src;
        dst.file.set(inputVariable,map);
    }

    Template expandVotes(const Template &t) const {
        // Create a new template matching the one containing the votes in the map structure
        // but remove the map structure
        Template expandedT = t;
        expandedT.file.remove(inputVariable);

        QMap<QString,QVariant> map = t.file.get<QMap<QString,QVariant> >(inputVariable);
        QMapIterator<QString, QVariant> i(map);
        bool ok;

        while (i.hasNext()) {
            i.next();
            // Normalize to [-1,1]
            float value = i.value().toFloat(&ok)/maxVotes;//* 2./maxVotes - 1;
            if (!ok) qFatal("Failed to expand Turk votes for %s", inputVariable);
            if (classify) (value > 0) ? value = 1 : value = -1;
            else if (consensusOnly && (value != 1 && value != -1)) continue;
            expandedT.file.set(i.key(),value);
        }

        return expandedT;
    }
};

BR_REGISTER(Transform, TurkTransform)*/

} // namespace br

#include "template.moc"
