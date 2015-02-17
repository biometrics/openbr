#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup distances
 * \brief Unmaps Turk HITs to be compared against query mats
 * \author Scott Klum \cite sklum
 */
class TurkDistance : public UntrainableDistance
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key)
    Q_PROPERTY(QStringList values READ get_values WRITE set_values RESET reset_values STORED false)
    BR_PROPERTY(QString, key, QString())
    BR_PROPERTY(QStringList, values, QStringList())

    bool targetHuman;
    bool queryMachine;

    void init()
    {
        targetHuman = Globals->property("TurkTargetHuman").toBool();
        queryMachine = Globals->property("TurkQueryMachine").toBool();
    }

    cv::Mat getValues(const Template &t) const
    {
        QList<float> result;
        foreach (const QString &value, values)
            result.append(t.file.get<float>(key + "_" + value));
        return OpenCVUtils::toMat(result, 1);
    }

    float compare(const Template &target, const Template &query) const
    {
        const cv::Mat a = targetHuman ? getValues(target) : target.m();
        const cv::Mat b = queryMachine ? query.m() : getValues(query);
        return -norm(a, b, cv::NORM_L1);
    }
};

BR_REGISTER(Distance, TurkDistance)

} // namespace br

#include "distance/turk.moc"
