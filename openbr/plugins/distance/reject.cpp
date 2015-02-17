#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Sets distance to -FLOAT_MAX if a target template has/doesn't have a key.
 * \author Scott Klum \cite sklum
 */
class RejectDistance : public UntrainableDistance
{
    Q_OBJECT

    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    BR_PROPERTY(QStringList, keys, QStringList())
    Q_PROPERTY(bool rejectIfContains READ get_rejectIfContains WRITE set_rejectIfContains RESET reset_rejectIfContains STORED false)
    BR_PROPERTY(bool, rejectIfContains, false)

    float compare(const Template &a, const Template &b) const
    {
        // We don't look at the query
        (void) b;

        foreach (const QString &key, keys)
            if ((rejectIfContains && a.file.contains(key)) || (!rejectIfContains && !a.file.contains(key)))
                return -std::numeric_limits<float>::max();

        return 0;
    }
};


BR_REGISTER(Distance, RejectDistance)

} // namespace br

#include "distance/reject.moc"
