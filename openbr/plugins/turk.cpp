#include "openbr_internal.h"
#include "openbr/core/common.h"
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

        Attribute normalized() const
        {
            bool ok;
            QList<float> values;
            foreach (const QString &value, *this) {
                values.append(value.toFloat(&ok));
                if (!ok)
                    qFatal("Can't normalize non-numeric vector!");
            }

            Attribute normal(name);
            float sum = Common::Sum(values);
            if (sum == 0) sum = 1;
            for (int i=0; i<values.size(); i++)
                normal.append(QString::number(values[i] / sum));
            return normal;
        }
    };

    TemplateList readBlock(bool *done)
    {
        *done = true;
        QStringList lines = QtUtils::readLines(file);
        QList<Attribute> headers;
        if (!lines.isEmpty())
            foreach (const QString &header, parse(lines.takeFirst()))
                headers.append(header);

        TemplateList templates;
        foreach (const QString &line, lines) {
            QStringList words = parse(line);
            if (words.size() != headers.size())
                qFatal("turkGallery invalid column count");

            File f;
            f.name = words[0];
            f.set("Label", words[0].mid(0,5));

            for (int i=1; i<words.size(); i++) {
                Attribute ratings = Attribute(words[i]).normalized();
                if (headers[i].size() != ratings.size())
                    qFatal("turkGallery invalid attribute count");
                for (int j=0; j<ratings.size(); j++)
                    f.set(headers[i].name + "_" + headers[i][j], ratings[j]);
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
    Q_PROPERTY(bool isMeta READ get_isMeta WRITE set_isMeta RESET reset_isMeta STORED false)
    BR_PROPERTY(QString, key, QString())
    BR_PROPERTY(QStringList, values, QStringList())
    BR_PROPERTY(bool, isMeta, false)

    Transform *child;

    void init()
    {
        QStringList classifiers;
        foreach (const QString &value, values) 
            classifiers.append(QString("SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=%1,outputVariable=predicted_%1)%2").arg(key + "_" + value).arg(isMeta ? QString("+Average+SaveMat(predicted_%1)").arg(value) : ""));
        QString algorithm = classifiers.join("/") + (classifiers.size() > 1 ? "+Cat" : "");
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
 * \ingroup distances
 * \brief Unmaps Turk HITs to be compared against query mats
 * \author Scott Klum \cite sklum
 */
class TurkDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key)
    Q_PROPERTY(QStringList values READ get_values WRITE set_values RESET reset_values STORED false)
    BR_PROPERTY(QString, key, QString())
    BR_PROPERTY(QStringList, values, QStringList())

    float compare(const Template &target, const Template &query) const
    {
        const float stddev = .75;
        float score = 0;
        for (int i=0; i<values.size(); i++)
            score += 1 / (stddev*sqrt(2*CV_PI)) * exp(-0.5*pow((query.m().at<float>(0,i)-target.file.get<float>(key + "_" + values[i]))/stddev, 2));
        return score;
    }
};

BR_REGISTER(Distance, TurkDistance)

} // namespace br

#include "turk.moc"
