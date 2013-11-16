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
        const QRegularExpressionMatch match = re.match(key.isEmpty() ? src.file.suffix() : src.file.get<QString>(key));
        if (match.hasMatch()) dst = Template();
        else                  dst = src;
    }
};

BR_REGISTER(Transform, RemoveTemplatesTransform)

/*!
 * \ingroup transforms
 * \brief Name a point
 * \author Scott Klum \cite sklum
 */
class NamePointsTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    Q_PROPERTY(QStringList names READ get_names WRITE set_names RESET reset_names STORED false)
    BR_PROPERTY(QList<int>, indices, QList<int>())
    BR_PROPERTY(QStringList, names, QStringList())

    void project(const Template &src, Template &dst) const
    {
        if (indices.size() != names.size()) qFatal("Point/name size mismatch");

        dst = src;

        QList<QPointF> points = src.file.points();

        for (int i=0; i<indices.size(); i++) {
            if (indices[i] < points.size()) dst.file.set(names[i], points[indices[i]]);
            else qFatal("Idex out of range.");
        }
    }
};

BR_REGISTER(Transform, NamePointsTransform)

/*!
 * \ingroup transforms
 * \brief Remove a name from a point
 * \author Scott Klum \cite sklum
 */
class AnonymizePointsTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList names READ get_names WRITE set_names RESET reset_names STORED false)
    BR_PROPERTY(QStringList, names, QStringList())

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        foreach (const QString &name, names)
            if (src.file.contains(name)) dst.file.appendPoint(src.file.get<QPointF>(name));
    }
};

BR_REGISTER(Transform, AnonymizePointsTransform)

} // namespace br

#include "template.moc"
