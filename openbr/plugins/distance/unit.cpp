#include <openbr/plugins/openbr_internal.h>

namespace br
{

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
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    BR_PROPERTY(br::Distance*, distance, make("Dist(L2)"))
    BR_PROPERTY(float, a, 1)
    BR_PROPERTY(float, b, 0)
    BR_PROPERTY(QString, inputVariable, "Label")

    void train(const TemplateList &templates)
    {
        const TemplateList samples = templates.mid(0, 2000);
        const QList<int> sampleLabels = samples.indexProperty(inputVariable);
        QScopedPointer<MatrixOutput> matrixOutput(MatrixOutput::make(FileList(samples.size()), FileList(samples.size())));
        Distance::compare(samples, samples, matrixOutput.data());

        double genuineAccumulator, impostorAccumulator;
        int genuineCount, impostorCount;
        genuineAccumulator = impostorAccumulator = genuineCount = impostorCount = 0;

        for (int i=0; i<samples.size(); i++) {
            for (int j=0; j<i; j++) {
                const float val = matrixOutput.data()->data.at<float>(i, j);
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

} // namespace br

#include "distance/unit.moc"
