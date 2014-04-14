#include "openbr_internal.h"
#include "openbr/core/qtutils.h"

namespace br
{

/*!
 * \ingroup galleries
 * \brief For Amazon Mechanical Turk datasets
 * \author Scott Klum \cite sklum
 */
class turkGallery : public Gallery
{
    Q_OBJECT

    struct Attribute : public QStringList
    {
        QString name;
        Attribute(const QString &str = QString())
        {
            const int i = str.indexOf('[');
            name = str.mid(0, i);
            if (i != -1)
                append(str.mid(i+1, str.length()-i-2).split(","));
        }
    };

    TemplateList readBlock(bool *done)
    {
        *done = true;
        TemplateList templates;
        QStringList lines = QtUtils::readLines(file);
        if (lines.empty())
            qFatal(".turk Gallery missing header.");
        QList<Attribute> types;
        foreach (const QString &header, parse(lines.takeFirst()))
            types.append(header);

        foreach (const QString &line, lines) {
            const QStringList words = parse(line);
            if (words.size() != types.size())
                qFatal(".turk Gallery incorrect column count.");

            File f(words[0], words[0].mid(0,5));
            for (int i=1; i<words.size(); i++) {
                Attribute &type = types[i];
                Attribute rating(words[i]);
                if (type.size() != rating.size())
                    qFatal(".turk Gallery incorrect ratings count.");

                QMap<QString,QVariant> categoryMap;
                for (int j=0; j<type.size(); j++)
                    categoryMap.insert(type[j], rating[j]);
                f.set(type.name, categoryMap);
            }
            templates.append(f);
        }

        return templates;
    }

    void write(const Template &)
    {
        qFatal("turkGallery write not implemented.");
    }
};

BR_REGISTER(Gallery, turkGallery)

static Template unmap(const Template &t, const QString& variable, const float maxVotes, const float maxRange, const float minRange, const bool classify, const bool consensusOnly) {
    // Create a new template matching the one containing the votes in the map structure
    // but remove the map structure
    Template expandedT = t;
    expandedT.file.remove(variable);

    QMap<QString,QVariant> map = t.file.get<QMap<QString,QVariant> >(variable);
    QMapIterator<QString, QVariant> i(map);
    bool ok;

    while (i.hasNext()) {
        i.next();
        // Normalize to [minRange,maxRange]
        float value = i.value().toFloat(&ok)*(maxRange-minRange)/maxVotes - minRange;
        if (!ok) qFatal("Failed to expand Turk votes for %s", qPrintable(variable));
        if (classify) (value > maxRange-((maxRange-minRange)/2)) ? value = maxRange : value = minRange;
        else if (consensusOnly && (value != maxRange && value != minRange)) continue;
        expandedT.file.set(i.key(),value);
    }

    return expandedT;
}

/*!
 * \ingroup transforms
 * \brief Converts Amazon MTurk labels to a non-map format for use in a transform
 * \author Scott Klum \cite sklum
 */
class TurkTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString HIT READ get_HIT WRITE set_HIT RESET reset_HIT STORED false)
    Q_PROPERTY(float maxVotes READ get_maxVotes WRITE set_maxVotes RESET reset_maxVotes STORED false)
    Q_PROPERTY(float maxRange READ get_maxRange WRITE set_maxRange RESET reset_maxRange STORED false)
    Q_PROPERTY(float minRange READ get_minRange WRITE set_minRange RESET reset_minRange STORED false)
    Q_PROPERTY(bool classify READ get_classify WRITE set_classify RESET reset_classify STORED false)
    Q_PROPERTY(bool consensusOnly READ get_consensusOnly WRITE set_consensusOnly RESET reset_consensusOnly STORED false)
    BR_PROPERTY(QString, HIT, QString())
    BR_PROPERTY(float, maxVotes, 1)
    BR_PROPERTY(float, maxRange, 1)
    BR_PROPERTY(float, minRange, 0)
    BR_PROPERTY(bool, classify, false)
    BR_PROPERTY(bool, consensusOnly, false)

    void project(const Template &src, Template &dst) const
    {
        dst = unmap(src, HIT, maxVotes, maxRange, minRange, classify, consensusOnly);
    }
};

BR_REGISTER(Transform, TurkTransform)

/*!
 * \ingroup transforms
 * \brief Convenience class for training turk attribute regressors
 * \author Josh Klontz \cite jklontz
 */
class TurkClassifierTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    Q_PROPERTY(QStringList values READ get_values WRITE set_values RESET reset_values STORED false)
    Q_PROPERTY(float maxVotes READ get_maxVotes WRITE set_maxVotes RESET reset_maxVotes STORED false)
    BR_PROPERTY(QString, key, QString())
    BR_PROPERTY(QStringList, values, QStringList())
    BR_PROPERTY(float, maxVotes, 1)

    Transform *child;

    void init()
    {
        QString algorithm = QString("Turk(%1, %2)+").arg(key, QString::number(maxVotes));
        QStringList classifiers;
        foreach (const QString &value, values)
            classifiers.append(QString("SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=%1,outputVariable=predicted_%1)").arg(value));
        algorithm += classifiers.join("/");
        if (values.size() > 1)
            algorithm += "+Cat";
        child = Transform::make(algorithm);
    }

    void train(const QList<TemplateList> &data)
    {
        child->train(data);
    }

    void project(const Template &src, Template &dst) const
    {
        child->project(src, dst);
    }

    void store(QDataStream &stream) const
    {
        child->store(stream);
    }

    void load(QDataStream &stream)
    {
        child->load(stream);
    }
};

BR_REGISTER(Transform, TurkClassifierTransform)

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
            if (t.file.contains(s)) {
                map.insert(s,t.file.value(s));
                mappedT.file.remove(s);
            }
        }

        if (!map.isEmpty()) mappedT.file.set(outputVariable,map);

        return mappedT;
    }
};

BR_REGISTER(Transform, MapTransform)


/*!
 * \ingroup distances
 * \brief Unmaps Turk HITs to be compared against query mats
 * \author Scott Klum \cite sklum
 */
class TurkDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(QString HIT READ get_HIT WRITE set_HIT RESET reset_HIT)
    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    Q_PROPERTY(float maxVotes READ get_maxVotes WRITE set_maxVotes RESET reset_maxVotes STORED false)
    Q_PROPERTY(float maxRange READ get_maxRange WRITE set_maxRange RESET reset_maxRange STORED false)
    Q_PROPERTY(float minRange READ get_minRange WRITE set_minRange RESET reset_minRange STORED false)
    Q_PROPERTY(bool classify READ get_classify WRITE set_classify RESET reset_classify STORED false)
    Q_PROPERTY(bool consensusOnly READ get_consensusOnly WRITE set_consensusOnly RESET reset_consensusOnly STORED false)
    BR_PROPERTY(QString, HIT, QString())
    BR_PROPERTY(QStringList, keys, QStringList())
    BR_PROPERTY(float, maxVotes, 1)
    BR_PROPERTY(float, maxRange, 1)
    BR_PROPERTY(float, minRange, 0)
    BR_PROPERTY(bool, classify, false)
    BR_PROPERTY(bool, consensusOnly, false)

    float compare(const Template &target, const Template &query) const
    {
        Template t = unmap(target, HIT, maxVotes, maxRange, minRange, classify, consensusOnly);

        QList<float> targetValues;
        foreach(const QString &s, keys) targetValues.append(t.file.get<float>(s));

        float stddev = .75;

        float score = 0;
        for (int i=0; i<targetValues.size(); i++) score += 1/(stddev*sqrt(2*CV_PI))*exp(-0.5*pow((query.m().at<float>(0,i)-targetValues[i])/stddev, 2));

        return score;
    }
};

BR_REGISTER(Distance, TurkDistance)

} // namespace br

#include "turk.moc"
