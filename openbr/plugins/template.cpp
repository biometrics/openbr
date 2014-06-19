#include <QtCore>

#include "openbr_internal.h"

namespace br
{

/*!
 * \ingroup transforms
 * \brief Retains only the values for the keys listed, to reduce template size
 * \author Scott Klum \cite sklum
 */
class KeepMetadataTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    BR_PROPERTY(QStringList, keys, QStringList())

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        foreach(const QString& localKey, dst.localKeys()) {
            if (!keys.contains(localKey)) dst.remove(localKey);
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
 * \brief Sets the metadata key/value pair.
 * \author Josh Klontz \cite jklontz
 */
class SetMetadataTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    Q_PROPERTY(QString value READ get_value WRITE set_value RESET reset_value STORED false)
    BR_PROPERTY(QString, key, "")
    BR_PROPERTY(QString, value, "")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        dst.set(key, value);
    }
};

BR_REGISTER(Transform, SetMetadataTransform)

/*!
 * \ingroup transforms
 * \brief Clear templates without the required metadata.
 * \author Josh Klontz \cite jklontz
 */
class IfMetadataTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    Q_PROPERTY(QString value READ get_value WRITE set_value RESET reset_value STORED false)
    BR_PROPERTY(QString, key, "")
    BR_PROPERTY(QString, value, "")

    void project(const Template &src, Template &dst) const
    {
        if (src.file.get<QString>(key, "") == value)
            dst = src;
    }
};

BR_REGISTER(Transform, IfMetadataTransform)

/*!
 * \ingroup transforms
 * \brief Represent the metadata as JSON template data.
 * \author Josh Klontz \cite jklontz
 */
class JSONTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        const QByteArray json = QJsonDocument(QJsonObject::fromVariantMap(src.file.localMetadata())).toJson();
        dst += cv::Mat(1, json.size(), CV_8UC1, (void*) json.data());
    }
};

BR_REGISTER(Transform, JSONTransform)

/*!
 * \ingroup transforms
 * \brief Removes a metadata field from all templates
 * \author Brendan Klare \cite bklare
 */
class RemoveMetadataTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QString attributeName READ get_attributeName WRITE set_attributeName RESET reset_attributeName STORED false)
    BR_PROPERTY(QString, attributeName, "None")

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        if (dst.contains(attributeName))
            dst.remove(attributeName);
    }
};
BR_REGISTER(Transform, RemoveMetadataTransform)

/*!
 * \ingroup transforms
 * \brief Retains only landmarks/points at the provided indices
 * \author Brendan Klare \cite bklare
 */
class SelectPointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    BR_PROPERTY(QList<int>, indices, QList<int>())

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        QList<QPointF> origPoints = src.points();
        dst.clearPoints();
        for (int i = 0; i < indices.size(); i++)
            dst.appendPoint(origPoints[indices[i]]);
    }
};

BR_REGISTER(Transform, SelectPointsTransform)

} // namespace br

#include "template.moc"
