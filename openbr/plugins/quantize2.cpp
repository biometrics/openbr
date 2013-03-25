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

    static void computeThresholds(const Mat &data, const QList<int> &labels, float *thresholds)
    {
        const QList<float> vals = OpenCVUtils::matrixToVector<float>(data);
        if (vals.size() != labels.size())
            qFatal("Logic error.");

        QList<float> genuineScores; genuineScores.reserve(vals.size());
        QList<float> impostorScores; impostorScores.reserve(vals.size()*vals.size()/2);
        for (int i=0; i<vals.size(); i++)
            for (int j=i+1; j<vals.size(); j++)
                if (labels[i] == labels[j]) genuineScores.append(fabs(vals[i]-vals[j]));
                else                        impostorScores.append(fabs(vals[i]-vals[j]));

       // genuineScores = Common::Downsample(genuineScores, 256);
        impostorScores = Common::Downsample(impostorScores, genuineScores.size());
        double hGenuine = Common::KernelDensityBandwidth(genuineScores);
        double hImpostor = Common::KernelDensityBandwidth(impostorScores);

        float genuineMin, genuineMax, impostorMin, impostorMax, min, max;
        Common::MinMax(genuineScores, &genuineMin, &genuineMax);
        Common::MinMax(impostorScores, &impostorMin, &impostorMax);
        min = std::min(genuineMin, impostorMin);
        max = std::max(genuineMax, impostorMax);
        qDebug() << genuineMin << genuineMax << impostorMin << impostorMax;

        QFile g("g.csv"), i("i.csv"), kde("kde.csv");
        g.open(QFile::Append); i.open(QFile::Append); kde.open(QFile::Append);

        QStringList words;
        const int steps = 256;
        for (int i=0; i<steps; i++) {
            const float score = min + i*(max-min)/(steps-1);
            words.append(QString::number(log(Common::KernelDensityEstimation(genuineScores, score, hGenuine)/
                                             Common::KernelDensityEstimation(impostorScores, score, hImpostor))));
        }

        g.write(qPrintable(QtUtils::toStringList(genuineScores).join(",")+"\n"));
        i.write(qPrintable(QtUtils::toStringList(impostorScores).join(",")+"\n"));
        kde.write(qPrintable(words.join(",")+"\n"));
        g.close(); i.close(); kde.close();
        abort();
    }

    void train(const TemplateList &src)
    {
        const Mat data = OpenCVUtils::toMat(src.data());
        const QList<int> labels = src.labels<int>();

        thresholds = QVector<float>(256*data.cols);

        QFutureSynchronizer<void> futures;
        for (int i=0; i<data.cols; i++)
            if (false) futures.addFuture(QtConcurrent::run(&BayesianQuantizationTransform::computeThresholds, data.col(i), labels, &thresholds.data()[i*256]));
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
