#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>

#include <QtConcurrent>

using namespace cv;

namespace br
{

struct Miner
{
    Mat src;
    Mat scaledSrc;
    Size windowSize;
    Point offset, point;
    float scale, scaleFactor, stepFactor;

    Miner(const Mat &m, const Size &windowSize, const Point &offset) :
        src(m),
        windowSize(windowSize),
        offset(offset),
        point(offset)
    {
        scale       = 1.0F;
        scaleFactor = 1.4142135623730950488016887242097F;
        stepFactor  = 0.5F;

        scale = max(((float)windowSize.width + point.x) / ((float)src.cols),
                    ((float)windowSize.height + point.y) / ((float)src.rows));
        Size size((int)(scale*src.cols + 0.5F), (int)(scale*src.rows + 0.5F));
        resize(src, scaledSrc, size);
    }

    Mat mine(bool *newImg)
    {
        // Copy region of winSize region of img into m
        Mat window(windowSize.height, windowSize.width, CV_8UC1,
                   (void*)(scaledSrc.data + point.y * scaledSrc.step + point.x * scaledSrc.elemSize()),
                   scaledSrc.step);

        Mat sample;
        window.copyTo(sample);

        if ((int)(point.x + (1.0F + stepFactor) * windowSize.width) < scaledSrc.cols)
            point.x += (int)(stepFactor * windowSize.width);
        else {
            point.x = offset.x;
            if ((int)(point.y + (1.0F + stepFactor) * windowSize.height) < scaledSrc.rows)
                point.y += (int)(stepFactor * windowSize.height);
            else {
                point.y = offset.y;
                scale *= scaleFactor;
                if (scale <= 1.0F) {
                    Size size((int)(scale*src.cols), (int)(scale*src.rows));
                    resize(src, scaledSrc, size);
                } else {
                    *newImg = true;
                    return sample;
                }
            }
        }

        *newImg = false;
        return sample;
    }
};

/*!
 * \brief A meta Classifier that creates a cascade of another Classifier. The cascade is a series of stages, each with its own instance of a given classifier. A sample can only reach the next stage if it is classified as positive by the previous stage.
 * \author Jordan Cheney \cite jcheney
 * \author Scott Klum \cite sklum
 * \br_property int numStages The number of stages in the cascade
 * \br_property int numPos The number of positives to feed each stage during training
 * \br_property int numNegs The number of negatives to feed each stage during training. A negative sample must have been classified by the previous stages in the cascade as positive to be fed to the next stage during training.
 * \br_property float maxFAR A termination parameter. Calculated as (number of passed negatives) / (total number of checked negatives) for a given stage during training. If that number is below the given maxFAR cascade training is terminated early. This can help prevent overfitting.
 * \br_paper Paul Viola, Michael Jones
 *           Rapid Object Detection using a Boosted Cascade of Simple Features
 *           CVPR, 2001
 * \br_link Rapid Object Detection using a Boosted Cascade of Simple Features https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
 */
class CascadeClassifier : public Classifier
{
    Q_OBJECT

    Q_PROPERTY(QString stageDescription READ get_stageDescription WRITE set_stageDescription RESET reset_stageDescription STORED false)
    Q_PROPERTY(int numStages READ get_numStages WRITE set_numStages RESET reset_numStages STORED false)
    Q_PROPERTY(int numPos READ get_numPos WRITE set_numPos RESET reset_numPos STORED false)
    Q_PROPERTY(int numNegs READ get_numNegs WRITE set_numNegs RESET reset_numNegs STORED false)
    Q_PROPERTY(float maxFAR READ get_maxFAR WRITE set_maxFAR RESET reset_maxFAR STORED false)
    Q_PROPERTY(bool requireAllStages READ get_requireAllStages WRITE set_requireAllStages RESET reset_requireAllStages STORED false)

    BR_PROPERTY(QString, stageDescription, "")
    BR_PROPERTY(int, numStages, 20)
    BR_PROPERTY(int, numPos, 1000)
    BR_PROPERTY(int, numNegs, 1000)
    BR_PROPERTY(float, maxFAR, pow(0.5, numStages))
    BR_PROPERTY(bool, requireAllStages, false)

    QList<Classifier *> stages;
    TemplateList posImages, negImages;
    TemplateList posSamples, negSamples;

    QList<int> indices;
    int negIndex, posIndex, samplingRound;

    QMutex samplingMutex, miningMutex;

    void init()
    {
        negIndex = posIndex = samplingRound = 0;
    }

    bool getPositive(Template &img)
    {
        if (posIndex >= posImages.size())
            return false;
        img = posImages[indices[posIndex++]];
        return true;
    }

    Template getNegative(Point &offset)
    {
        Template negative;

        const Size size = windowSize();
        // Grab negative from list
        int count = negImages.size();
        for (int i = 0; i < count; i++) {
            negative = negImages[negIndex++];

            samplingRound += negIndex / count;
            samplingRound = samplingRound % (size.width * size.height);
            negIndex %= count;

            offset.x = qMin( (int)samplingRound % size.width, negative.m().cols - size.width);
            offset.y = qMin( (int)samplingRound / size.width, negative.m().rows - size.height);
            if (!negative.m().empty() && negative.m().type() == CV_8UC1
                    && offset.x >= 0 && offset.y >= 0)
                break;
        }

        return negative;
    }

    uint64 mine()
    {
        uint64 passedNegatives = 0;
        forever {
            Template negative;
            Point offset;
            QMutexLocker samplingLocker(&samplingMutex);
            negative = getNegative(offset);
            samplingLocker.unlock();

            Miner miner(negative.m(), windowSize(), offset);
            forever {
                bool newImg;
                Template sample(negative.file, miner.mine(&newImg));
                if (!newImg) {
                    if (negSamples.size() >= numNegs)
                        return passedNegatives;

                    float confidence;
                    if (classify(sample, true, &confidence) != 0) {
                        QMutexLocker miningLocker(&miningMutex);
                        if (negSamples.size() >= numNegs)
                            return passedNegatives;

                        negSamples.append(sample);
                        printf("Negative samples: %d\r", negSamples.size());
                    }

                    passedNegatives++;
                } else
                    break;
            }
        }
    }

    void train(const TemplateList &data)
    {
        foreach (const Template &t, data)
            t.file.get<float>("Label") == 1.0f ? posImages.append(t) : negImages.append(t);

        qDebug() << "Total images:" << data.size()
                 << "\nTotal positive images:" << posImages.size()
                 << "\nTotal negative images:" << negImages.size();

        indices = Common::RandSample(posImages.size(), posImages.size(), true);

        stages.reserve(numStages);
        for (int i = 0; i < numStages; i++) {
            Classifier *next_stage = Classifier::make(stageDescription, NULL);
            stages.append(next_stage);
        }

        for (int i = 0; i < numStages; i++) {
            qDebug() << "===== TRAINING" << i << "stage =====";
            qDebug() << "<BEGIN";

            float currFAR = getSamples();

            if (currFAR < maxFAR && !requireAllStages) {
                qDebug() << "FAR is below required level! Terminating early";
                return;
            }

            stages[i]->train(posSamples + negSamples);

            qDebug() << "END>";
        }
    }

    float classify(const Template &src, bool process, float *confidence) const
    {
        float stageConf = 0.0f;
        foreach (const Classifier *stage, stages) {
            float result = stage->classify(src, process, &stageConf);
            if (confidence)
                *confidence += stageConf;
            if (result == 0.0f)
                return 0.0f;
        }
        return 1.0f;
    }

    int numFeatures() const
    {
        return stages.first()->numFeatures();
    }

    Template preprocess(const Template &src) const
    {
        return stages.first()->preprocess(src);
    }

    Size windowSize(int *dx = NULL, int *dy = NULL) const
    {
        return stages.first()->windowSize(dx, dy);
    }

    void load(QDataStream &stream)
    {
        int numStages; stream >> numStages;
        for (int i = 0; i < numStages; i++) {
            Classifier *nextStage = Classifier::make(stageDescription, NULL);
            nextStage->load(stream);
            stages.append(nextStage);
        }
    }

    void store(QDataStream &stream) const
    {
        stream << stages.size();
        foreach (const Classifier *stage, stages)
            stage->store(stream);
    }

private:
    float getSamples()
    {
        posSamples.clear(); posSamples.reserve(numPos);
        negSamples.clear(); negSamples.reserve(numNegs);
        posIndex = 0;

        float confidence;
        while (posSamples.size() < numPos) {
            Template pos;
            if (!getPositive(pos))
                qFatal("Cannot get another positive sample!");

            if (classify(pos, true, &confidence) > 0.0f) {
                printf("POS current samples: %d\r", posSamples.size());
                posSamples.append(pos);
            }
        }

        qDebug() << "POS count : consumed  " << posSamples.size() << ":" << posIndex;

        QFutureSynchronizer<uint64> futures;
        for (int i=0; i<QThread::idealThreadCount(); i++)
            futures.addFuture(QtConcurrent::run(this, &CascadeClassifier::mine));
        futures.waitForFinished();

        uint64 passedNegs = 0;
        QList<QFuture<uint64> > results = futures.futures();
        for (int i=0; i<results.size(); i++)
            passedNegs += results[i].result();

        double acceptanceRatio = negSamples.size() / (double)passedNegs;
        qDebug() << "NEG count : acceptanceRatio  " << negSamples.size() << ":" << acceptanceRatio;

        return acceptanceRatio;
    }
};

BR_REGISTER(Classifier, CascadeClassifier)

} // namespace br

#include "classification/cascade.moc"
