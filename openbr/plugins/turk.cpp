#include "openbr_internal.h"

namespace br
{

/*!
 * \ingroup transforms
 * \brief Converts Amazon MTurk labels to a non-map format for use in a transform
 *        Also optionally normalizes and/or classifies the votes
 * \author Scott Klum \cite sklum
 */
class TurkTransform : public UntrainableTransform
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
        dst = unmap(src);
    }

    Template unmap(const Template &t) const {
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

BR_REGISTER(Transform, TurkTransform)

/*!
 * \ingroup transforms
 * \brief Converts metadata into a map structure
 * \author Scott Klum \cite sklum
 */
class MapTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList inputVariables READ get_inputVariables WRITE set_inputVariables RESET reset_inputVariables STORED false)
    Q_PROPERTY(QString outputVariable READ get_outputVariable WRITE set_outputVariable RESET reset_outputVariable STORED false)
    BR_PROPERTY(QStringList, inputVariables, QStringList())
    BR_PROPERTY(QString, outputVariable, QString())

    void project(const Template &src, Template &dst) const
    {
        dst = map(src);
    }

    Template map(const Template &t) const {
        Template mappedT = t;
        QMap<QString,QVariant> map;

        foreach(const QString &s, inputVariables) {
            // Get checks if the variant stored in m_metdata can be
            // converted to the type T. For some reason, you cannot
            // convert from a QVariant to a QVariant.  Thus, this transform
            // has to assume that the metadata we want to organize can be
            // converted to a float, resulting in a loss of generality :(.
            if (t.file.contains(s)) {
                map.insert(s,t.file.get<float>(s));
                mappedT.file.remove(s);
            }
        }

        if (!map.isEmpty()) mappedT.file.set(outputVariable,map);

        return mappedT;
    }
};

BR_REGISTER(Transform, MapTransform)

} // namespace br

#include "turk.moc"
