#include <QtCore>

#include "openbr_internal.h"
#include "openbr/core/common.h"

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
        foreach (const QString& localKey, dst.localKeys())
            if (!keys.contains(localKey))
                dst.remove(localKey);
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
        dst.file.set("AlgorithmID", 2);
        const QByteArray json = QJsonDocument(QJsonObject::fromVariantMap(dst.file.localMetadata())).toJson().replace('\n', ' ');
        dst += cv::Mat(1, json.size()+1 /*include null terminator*/, CV_8UC1, (void*) json.data()).clone();
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
 * \brief Clears the points from a template
 * \author Brendan Klare \cite bklare
 */
class ClearPointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        dst.clearPoints();
    }
};

BR_REGISTER(Transform, ClearPointsTransform)

/*!
 * \ingroup transforms
 * \brief Retains only landmarks/points at the provided indices
 * \author Brendan Klare \cite bklare
 */
class SelectPointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    Q_PROPERTY(bool invert READ get_invert WRITE set_invert RESET reset_invert STORED false) // keep the points _not_ in the list
    BR_PROPERTY(QList<int>, indices, QList<int>())
    BR_PROPERTY(bool, invert, false)

    void projectMetadata(const File &src, File &dst) const
    {
        const QList<QPointF> srcPoints = src.points();
        QList<QPointF> dstPoints;
        for (int i=0; i<srcPoints.size(); i++)
            if (indices.contains(i) ^ invert)
                dstPoints.append(srcPoints[i]);
        dst.setPoints(dstPoints);
    }
};

BR_REGISTER(Transform, SelectPointsTransform)

/*!
 * \ingroup transforms
 * \brief Removes duplicate templates based on a unique metadata key
 * \author Austin Blanton \cite imaus10
 */
class FilterDupeMetadataTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    BR_PROPERTY(QString, key, "TemplateID")

    QSet<QString> excluded;

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        foreach (const Template &t, src) {
            QString id = t.file.get<QString>(key);
            if (!excluded.contains(id)) {
                dst.append(t);
                excluded.insert(id);
            }
        }
    }

public:
    FilterDupeMetadataTransform() : TimeVaryingTransform(false,false) {}
};

BR_REGISTER(Transform, FilterDupeMetadataTransform)

/*!
 * \ingroup transforms
 * \brief Reorder the points such that points[from[i]] becomes points[to[i]] and
 *        vice versa
 * \author Scott Klum \cite sklum
 */
class ReorderPointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    Q_PROPERTY(QList<int> from READ get_from WRITE set_from RESET reset_from STORED false)
    Q_PROPERTY(QList<int> to READ get_to WRITE set_to RESET reset_to STORED false)
    BR_PROPERTY(QList<int>, from, QList<int>())
    BR_PROPERTY(QList<int>, to, QList<int>())

    void projectMetadata(const File &src, File &dst) const
    {
        if (from.size() == to.size()) {
            QList<QPointF> points = src.points();
            int size = src.points().size();
            if (!points.contains(QPointF(-1,-1)) && Common::Max(from) < size && Common::Max(to) < size) {
                for (int i=0; i<from.size(); i++) {
                    std::swap(points[from[i]],points[to[i]]);
                }
                dst.setPoints(points);
            }
        } else qFatal("Inconsistent sizes for to and from index lists.");
    }
};

BR_REGISTER(Transform, ReorderPointsTransform)

} // namespace br

#include "template.moc"
