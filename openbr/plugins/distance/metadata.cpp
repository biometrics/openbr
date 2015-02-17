#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Checks target metadata against query metadata.
 * \author Scott Klum \cite sklum
 */
class MetadataDistance : public UntrainableDistance
{
    Q_OBJECT

    Q_PROPERTY(QStringList filters READ get_filters WRITE set_filters RESET reset_filters STORED false)
    BR_PROPERTY(QStringList, filters, QStringList())

    float compare(const Template &a, const Template &b) const
    {
        foreach (const QString &key, filters) {
            QString aValue = a.file.get<QString>(key, QString());
            QString bValue = b.file.get<QString>(key, QString());

            // The query value may be a range. Let's check.
            if (bValue.isEmpty()) bValue = QtUtils::toString(b.file.get<QPointF>(key, QPointF()));

            if (aValue.isEmpty() || bValue.isEmpty()) continue;

            bool keep = false;
            bool ok;

            QPointF range = QtUtils::toPoint(bValue,&ok);

            if (ok) /* Range */ {
                int value = range.x();
                int upperBound = range.y();

                while (value <= upperBound) {
                    if (aValue == QString::number(value)) {
                        keep = true;
                        break;
                    }
                    value++;
                }
            }
            else if (aValue == bValue) keep = true;

            if (!keep) return -std::numeric_limits<float>::max();
        }
        return 0;
    }
};


BR_REGISTER(Distance, MetadataDistance)

} // namespace br

#include "distance/metadata.moc"
