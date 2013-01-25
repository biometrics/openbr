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
    BR_PROPERTY(br::Distance*, distance, Distance::make("Dist(L2)", this))
    BR_PROPERTY(double, mean, 0)
    BR_PROPERTY(double, stddev, 1)
    br::TemplateList impostors;

    float calculateIUM(const Template &probe, const TemplateList &gallery) const
    {
        const int probeLabel = probe.file.label();
        TemplateList subset = gallery;
        for (int j=subset.size()-1; j>=0; j--)
            if (subset[j].file.label() == probeLabel)
                subset.removeAt(j);

        QList<float> scores = distance->compare(subset, probe);
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
        for (int i=0; i<data.size(); i++)
            iums.append(calculateIUM(impostors[i], impostors));

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

/* Kernel Density Estimator */
struct KDE
{
    float min, max;
    QList<float> bins;

    KDE() : min(0), max(1) {}
    KDE(const QList<float> &scores)
    {
        Common::MinMax(scores, &min, &max);
        double h = Common::KernelDensityBandwidth(scores);
        const int size = 255;
        bins.reserve(size);
        for (int i=0; i<size; i++)
            bins.append(Common::KernelDensityEstimation(scores, min + (max-min)*i/(size-1), h));
    }

    float operator()(float score) const
    {
        if (score <= min) return bins.first();
        if (score >= max) return bins.last();
        const float x = (score-min)/(max-min)*bins.size();
        const float y1 = bins[floor(x)];
        const float y2 = bins[ceil(x)];
        return y1 + (y2-y1)*(x-floor(x));
    }
};

QDataStream &operator<<(QDataStream &stream, const KDE &kde)
{
    return stream << kde.min << kde.max << kde.bins;
}

QDataStream &operator>>(QDataStream &stream, KDE &kde)
{
    return stream >> kde.min >> kde.max >> kde.bins;
}

/* Non-match Probability */
struct NMP
{
    KDE genuine, impostor;
    NMP() {}
    NMP(const QList<float> &genuineScores, const QList<float> &impostorScores)
        : genuine(genuineScores), impostor(impostorScores) {}
    float operator()(float score) const { float g = genuine(score); return g / (impostor(score) + g); }
};

QDataStream &operator<<(QDataStream &stream, const NMP &nmp)
{
    return stream << nmp.genuine << nmp.impostor;
}

QDataStream &operator>>(QDataStream &stream, NMP &nmp)
{
    return stream >> nmp.genuine >> nmp.impostor;
}

/*!
 * \ingroup distances
 * \brief Non-match Probability Distance \cite klare12
 * \author Josh Klontz \cite jklontz
 */
class NMPDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(QString binKey READ get_binKey WRITE set_binKey RESET reset_binKey STORED false)
    BR_PROPERTY(br::Distance*, distance, make("Dist(L2)"))
    BR_PROPERTY(QString, binKey, "")

    QHash<QString, NMP> nmps;

    void train(const TemplateList &src)
    {
        distance->train(src);

        const QList<int> labels = src.labels<int>();
        QScopedPointer<MatrixOutput> memoryOutput(dynamic_cast<MatrixOutput*>(Output::make(".Matrix", FileList(src.size()), FileList(src.size()))));
        distance->compare(src, src, memoryOutput.data());

        QHash< QString, QList<float> > genuineScores, impostorScores;
        for (int i=0; i<src.size(); i++)
            for (int j=0; j<i; j++) {
                const float score = memoryOutput.data()->data.at<float>(i, j);
                const QString bin = src[i].file.getString(binKey, "");
                if (labels[i] == labels[j]) genuineScores[bin].append(score);
                else                        impostorScores[bin].append(score);
            }

        foreach (const QString &key, genuineScores.keys())
            nmps.insert(key, NMP(genuineScores[key], impostorScores[key]));
    }

    float _compare(const Template &target, const Template &query) const
    {
        return nmps[query.file.getString(binKey, "")](distance->compare(target, query));
    }

    void store(QDataStream &stream) const
    {
        distance->store(stream);
        stream << nmps;
    }

    void load(QDataStream &stream)
    {
        distance->load(stream);
        stream >> nmps;
    }
};

BR_REGISTER(Distance, NMPDistance)

} // namespace br

#include "quality.moc"
