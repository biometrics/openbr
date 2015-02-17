#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>

namespace br
{

class ZScoreDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(bool crossModality READ get_crossModality WRITE set_crossModality RESET reset_crossModality STORED false)
    BR_PROPERTY(br::Distance*, distance, make("Dist(L2)"))
    BR_PROPERTY(bool, crossModality, false)

    float min, max;
    double mean, stddev;

    void train(const TemplateList &src)
    {
        distance->train(src);

        QScopedPointer<MatrixOutput> matrixOutput(MatrixOutput::make(FileList(src.size()), FileList(src.size())));
        distance->compare(src, src, matrixOutput.data());

        QList<float> scores;
        scores.reserve(src.size()*src.size());
        for (int i=0; i<src.size(); i++) {
            for (int j=0; j<i; j++) {
                const float score = matrixOutput.data()->data.at<float>(i, j);
                if (score == -std::numeric_limits<float>::max()) continue;
                if (crossModality && src[i].file.get<QString>("MODALITY") == src[j].file.get<QString>("MODALITY")) continue;
                scores.append(score);
            }
        }

        Common::MinMax(scores, &min, &max);
        Common::MeanStdDev(scores, &mean, &stddev);

        if (stddev == 0) qFatal("Stddev is 0.");
    }

    float compare(const Template &target, const Template &query) const
    {
        float score = distance->compare(target,query);
        if      (score == -std::numeric_limits<float>::max()) score = (min - mean) / stddev;
        else if (score ==  std::numeric_limits<float>::max()) score = (max - mean) / stddev;
        else                                                  score = (score - mean) / stddev;
        return score;
    }

    void store(QDataStream &stream) const
    {
        distance->store(stream);
        stream << min << max << mean << stddev;
    }

    void load(QDataStream &stream)
    {
        distance->load(stream);
        stream >> min >> max >> mean >> stddev;
    }
};

BR_REGISTER(Distance, ZScoreDistance)

} // namespace br

#include "distance/zscore.moc"
