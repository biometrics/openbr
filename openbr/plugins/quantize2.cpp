#include <QFutureSynchronizer>
#include <QtConcurrent>
#include <openbr/openbr_plugin.h>

#include "openbr/core/common.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/qtutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Quantize into a space where L1 distance approximates log-likelihood.
 * \author Josh Klontz \cite jklontz
 */
class BayesianQuantizationTransform : public Transform
{
    Q_OBJECT
    QVector<float> thresholds;

    static void computeThresholdsRecursive(const QVector<int> &cumulativeGenuines, const QVector<int> &cumulativeImpostors,
                                           float *thresholds,  const int thresholdIndex)
    {
//        const int totalGenuines = cumulativeGenuines.last()-cumulativeGenuines.first();
//        const int totalImpostors = cumulativeImpostors.last()-cumulativeImpostors.first();

        int low = 0;
        int high = cumulativeGenuines.size()-1;
        int index = cumulativeGenuines.size()/2;

        while ((index != low) && (index != high)) {
            index = (high - low)/2;
//            const float logLikelihoodLow = (float(cumulativeGenuines[index]-cumulativeGenuines.first())/totalGenuines)/
//                                           (float(cumulativeImpostors[index]-cumulativeImpostors.first())/totalImpostors);
//            const float logLikelihoodHigh = (float(cumulativeGenuines.last()-cumulativeGenuines[index])/totalGenuines)/
//                                            (float(cumulativeImpostors.last()-cumulativeImpostors[index])/totalImpostors);

        }

        computeThresholdsRecursive(cumulativeGenuines.mid(0,index), cumulativeImpostors.mid(0,index), thresholds, thresholdIndex);
        computeThresholdsRecursive(cumulativeGenuines.mid(index), cumulativeImpostors.mid(index), thresholds, thresholdIndex);
    }

    static void computeThresholds(const Mat &data, const QList<int> &labels, float *thresholds)
    {
        const QList<float> vals = OpenCVUtils::matrixToVector<float>(data);
        if (vals.size() != labels.size())
            qFatal("Logic error.");

        typedef QPair<float,bool> LabeledScore;
        QList<LabeledScore> labeledScores; labeledScores.reserve(vals.size());
        for (int i=0; i<vals.size(); i++)
            for (int j=i+1; j<vals.size(); j++)
                labeledScores.append(LabeledScore(fabs(vals[i]-vals[j]), labels[i] == labels[j]));
        std::sort(labeledScores.begin(), labeledScores.end());

        QVector<int> cumulativeGenuines(labeledScores.size());
        QVector<int> cumulativeImpostors(labeledScores.size());
        cumulativeGenuines[0] = (labeledScores.first().second ? 1 : 0);
        cumulativeImpostors[0] = (labeledScores.first().second ? 0 : 1);
        for (int i=1; i<labeledScores.size(); i++) {
            cumulativeGenuines[i] = cumulativeGenuines[i-1];
            cumulativeImpostors[i] = cumulativeImpostors[i-1];
            if (labeledScores.first().second) cumulativeGenuines[i]++;
            else                              cumulativeImpostors[i]++;
        }

        computeThresholdsRecursive(cumulativeGenuines, cumulativeImpostors, thresholds, 127);
    }

    void train(const TemplateList &src)
    {
        const Mat data = OpenCVUtils::toMat(src.data());
        const QList<int> labels = src.labels<int>();

        thresholds = QVector<float>(256*data.cols);

        QFutureSynchronizer<void> futures;
        for (int i=0; i<data.cols; i++)
            if (Globals->parallelism) futures.addFuture(QtConcurrent::run(&BayesianQuantizationTransform::computeThresholds, data.col(i), labels, &thresholds.data()[i*256]));
            else                                                                                          computeThresholds( data.col(i), labels, &thresholds.data()[i*256]);
        futures.waitForFinished();
    }

    void project(const Template &src, Template &dst) const
    {
        const QList<float> vals = OpenCVUtils::matrixToVector<float>(src);
        dst = Mat(1, vals.size(), CV_8UC1);
        for (int i=0; i<vals.size(); i++) {
            const float *t = &thresholds.data()[i*256];
            const float val = vals[i];
            uchar j = 0;
            while (val > t[j]) j++;
            dst.m().at<uchar>(0,i) = j;
        }
    }

    void store(QDataStream &stream) const
    {
        stream << thresholds;
    }

    void load(QDataStream &stream)
    {
        stream >> thresholds;
    }
};

BR_REGISTER(Transform, BayesianQuantizationTransform)

} // namespace br

#include "quantize2.moc"
