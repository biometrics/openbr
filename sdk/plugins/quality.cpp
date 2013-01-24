#include <openbr_plugin.h>

#include "core/common.h"

using namespace br;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Impostor Uniqueness Measure \cite klare12
 * \author Josh Klontz \cite jklontz
 */
class IUMTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(double mean READ get_mean WRITE set_mean RESET reset_mean)
    Q_PROPERTY(double stddev READ get_stddev WRITE set_stddev RESET reset_stddev)
    BR_PROPERTY(br::Distance*, distance, Factory<Distance>::make(".Dist(L2)"))
    BR_PROPERTY(double, mean, 0)
    BR_PROPERTY(double, stddev, 1)
    br::TemplateList impostors;

    float calculateIUM(const Template &probe, const TemplateList &gallery) const
    {
        QList<float> scores = distance->compare(gallery, probe);
        float min, max;
        Common::MinMax(scores, &min, &max);
        double mean;
        Common::Mean(scores, &mean);
        return (max-mean)/(max-min);
    }

    void train(const TemplateList &data)
    {
        distance->train(data);
        impostors = data;

        QList<float> iums; iums.reserve(impostors.size());
        QList<int> labels = impostors.labels<int>();
        for (int i=0; i<data.size(); i++) {
            TemplateList subset = impostors;
            for (int j=subset.size()-1; j>=0; j--)
                if (labels[j] == labels[i])
                    subset.removeAt(j);
            iums.append(calculateIUM(impostors[i], subset));
        }

        Common::MeanStdDev(iums, &mean, &stddev);
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        float ium = calculateIUM(src, impostors);
        dst.file.insert("IUM", ium);
        dst.file.insert("IUM_Bin", ium < mean-stddev ? 0 : (ium < mean+stddev ? 1 : 2));
    }

    void store(QDataStream &stream) const
    {
        distance->store(stream);
        stream << mean << stddev << impostors;
    }

    void load(QDataStream &stream)
    {
        distance->load(stream);
        stream >> mean >> stddev >> impostors;
    }
};

BR_REGISTER(Transform, IUMTransform)

} // namespace br

#include "quality.moc"
