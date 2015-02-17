#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Checks target metadata against filters.
 * \author Josh Klontz \cite jklontz
 */
class FilterDistance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        (void) b; // Query template isn't checked
        foreach (const QString &key, Globals->filters.keys()) {
            bool keep = false;
            const QString metadata = a.file.get<QString>(key, "");
            if (Globals->filters[key].isEmpty()) continue;
            if (metadata.isEmpty()) return -std::numeric_limits<float>::max();
            foreach (const QString &value, Globals->filters[key]) {
                if (metadata == value) {
                    keep = true;
                    break;
                }
            }
            if (!keep) return -std::numeric_limits<float>::max();
        }
        return 0;
    }
};

BR_REGISTER(Distance, FilterDistance)

} // namespace br

#include "distance/filter.moc"
