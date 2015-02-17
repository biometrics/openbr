#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Cross validate a distance metric.
 * \author Josh Klontz \cite jklontz
 */
class CrossValidateDistance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        static const QString key("Partition"); // More efficient to preallocate this
        const int partitionA = a.file.get<int>(key, 0);
        const int partitionB = b.file.get<int>(key, 0);
        return (partitionA != partitionB) ? -std::numeric_limits<float>::max() : 0;
    }
};

BR_REGISTER(Distance, CrossValidateDistance)

} // namespace br

#include "distance/crossvalidate.moc"
