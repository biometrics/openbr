#include <openbr_plugin.h>

#include "core/common.h"

namespace br
{

/*!
 * \ingroup transforms
 * \brief Impostor Uniqueness Measure \cite klare12
 * \author Josh Klontz \cite jklontz
 */
class ImpostorUniquenessMeasureTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(double mean READ get_mean WRITE set_mean RESET reset_mean)
    Q_PROPERTY(double stddev READ get_stddev WRITE set_stddev RESET reset_stddev)
    BR_PROPERTY(br::Distance*, distance, Distance::make("Dist(L2)", this))
    BR_PROPERTY(double, mean, 0)
    BR_PROPERTY(double, stddev, 1)
    TemplateList impostors;

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
        dst.file.insert("Impostor_Uniqueness_Measure", ium);
        dst.file.insert("Impostor_Uniqueness_Measure_Bin", ium < mean-stddev ? 0 : (ium < mean+stddev ? 1 : 2));
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

BR_REGISTER(Transform, ImpostorUniquenessMeasureTransform)

/* Kernel Density Estimator */
struct KDE
{
    float min, max;
    double mean, stddev;
    QList<float> bins;

    KDE() : min(0), max(1) {}
    KDE(const QList<float> &scores)
    {
        Common::MinMax(scores, &min, &max);
        Common::MeanStdDev(scores, &mean, &stddev);
        double h = Common::KernelDensityBandwidth(scores);
        const int size = 255;
        bins.reserve(size);
        for (int i=0; i<size; i++)
            bins.append(Common::KernelDensityEstimation(scores, min + (max-min)*i/(size-1), h));
    }

    float operator()(float score, bool gaussian = true) const
    {
        if (gaussian) return 1/(stddev*sqrt(2*CV_PI))*exp(-0.5*pow((score-mean)/stddev, 2));
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
    return stream << kde.min << kde.max << kde.mean << kde.stddev << kde.bins;
}

QDataStream &operator>>(QDataStream &stream, KDE &kde)
{
    return stream >> kde.min >> kde.max >> kde.mean >> kde.stddev >> kde.bins;
}

/* Match Probability */
struct MP
{
    KDE genuine, impostor;
    MP() {}
    MP(const QList<float> &genuineScores, const QList<float> &impostorScores)
        : genuine(genuineScores), impostor(impostorScores) {}
    float operator()(float score, bool gaussian = true) const
    {
        const float g = genuine(score, gaussian);
        const float s = g / (impostor(score, gaussian) + g);
        return s;
    }
};

QDataStream &operator<<(QDataStream &stream, const MP &nmp)
{
    return stream << nmp.genuine << nmp.impostor;
}

QDataStream &operator>>(QDataStream &stream, MP &nmp)
{
    return stream >> nmp.genuine >> nmp.impostor;
}

/*!
 * \ingroup distances
 * \brief Match Probability \cite klare12
 * \author Josh Klontz \cite jklontz
 */
class MatchProbabilityDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(QString binKey READ get_binKey WRITE set_binKey RESET reset_binKey STORED false)
    Q_PROPERTY(bool gaussian READ get_gaussian WRITE set_gaussian RESET reset_gaussian STORED false)
    BR_PROPERTY(br::Distance*, distance, make("Dist(L2)"))
    BR_PROPERTY(QString, binKey, "")
    BR_PROPERTY(bool, gaussian, true)

    QHash<QString, MP> mps;

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
                if (score == -std::numeric_limits<float>::max()) continue;
                const QString bin = src[i].file.getString(binKey, "");
                if (labels[i] == labels[j]) genuineScores[bin].append(score);
                else                        impostorScores[bin].append(score);
            }

        foreach (const QString &key, genuineScores.keys())
            mps.insert(key, MP(genuineScores[key], impostorScores[key]));
    }

    float compare(const Template &target, const Template &query) const
    {
        float rawScore = distance->compare(target, query);
        if (rawScore == -std::numeric_limits<float>::max()) return rawScore;
        return mps[query.file.getString(binKey, "")](rawScore, gaussian);
    }

    void store(QDataStream &stream) const
    {
        distance->store(stream);
        stream << mps;
    }

    void load(QDataStream &stream)
    {
        distance->load(stream);
        stream >> mps;
    }
};

BR_REGISTER(Distance, MatchProbabilityDistance)

/*!
 * \ingroup distances
 * \brief Linear normalizes of a distance so the mean impostor score is 0 and the mean genuine score is 1.
 * \author Josh Klontz \cite jklontz
 */
class UnitDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance *distance READ get_distance WRITE set_distance RESET reset_distance)
    Q_PROPERTY(float a READ get_a WRITE set_a RESET reset_a)
    Q_PROPERTY(float b READ get_b WRITE set_b RESET reset_b)
    BR_PROPERTY(br::Distance*, distance, make("Dist(L2)"))
    BR_PROPERTY(float, a, 1)
    BR_PROPERTY(float, b, 0)

    void train(const TemplateList &templates)
    {
        const TemplateList samples = templates.mid(0, 2000);
        const QList<float> sampleLabels = samples.labels<float>();
        QScopedPointer<MatrixOutput> memoryOutput(dynamic_cast<MatrixOutput*>(Output::make(".Matrix", FileList(samples.size()), FileList(samples.size()))));
        Distance::compare(samples, samples, memoryOutput.data());

        double genuineAccumulator, impostorAccumulator;
        int genuineCount, impostorCount;
        genuineAccumulator = impostorAccumulator = genuineCount = impostorCount = 0;

        for (int i=0; i<samples.size(); i++) {
            for (int j=0; j<i; j++) {
                const float val = memoryOutput.data()->data.at<float>(i, j);
                if (sampleLabels[i] == sampleLabels[j]) {
                    genuineAccumulator += val;
                    genuineCount++;
                } else {
                    impostorAccumulator += val;
                    impostorCount++;
                }
            }
        }

        if (genuineCount == 0) { qWarning("No genuine matches."); return; }
        if (impostorCount == 0) { qWarning("No impostor matches."); return; }

        double genuineMean = genuineAccumulator / genuineCount;
        double impostorMean = impostorAccumulator / impostorCount;

        if (genuineMean == impostorMean) { qWarning("Genuines and impostors are indistinguishable."); return; }

        a = 1.0/(genuineMean-impostorMean);
        b = impostorMean;

        qDebug("a = %f, b = %f", a, b);
    }

    float compare(const Template &target, const Template &query) const
    {
        return a * (distance->compare(target, query) - b);
    }
};

BR_REGISTER(Distance, UnitDistance)

/*!
 * \ingroup distances
 * \brief Check target metadata before matching templates.
 * \author Josh Klontz \cite jklontz
 */
class MetadataDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance *distance READ get_distance WRITE set_distance RESET reset_distance)
    BR_PROPERTY(br::Distance*, distance, make("Dist(L2)"))

    void train(const TemplateList &src)
    {
        distance->train(src);
    }

    float compare(const Template &a, const Template &b) const
    {
        foreach (const QString &filter, Globals->demographicFilters.keys()) {
            const QString metadata = a.file.getString(filter, "");
            if (metadata.isEmpty()) continue;
            const QRegExp re(Globals->demographicFilters[filter]);
            if (re.indexIn(metadata) == -1)
                return -std::numeric_limits<float>::max();
        }
        return distance->compare(a, b);
    }
};

BR_REGISTER(Distance, MetadataDistance)

} // namespace br

#include "quality.moc"
