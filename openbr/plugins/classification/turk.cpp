#include <openbr/plugins/openbr_internal.h>

namespace br
{

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
            classifiers.append(QString("(SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=%1,outputVariable=predicted_%1)%2)").arg(key + "_" + value, isMeta ? QString("+Average+SaveMat(predicted_%1)").arg(value) : QString()));
        child = Transform::make(classifiers.join("/") + (classifiers.size() > 1 ? "+Cat" : ""));
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

} // namespace br

#include "classification/turk.moc"
