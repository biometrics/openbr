#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>

using namespace cv;

namespace br
{

struct ImageHandler
{
    bool create(const QList<Mat> &_posImages, const QList<Mat> &_negImages, Size _winSize)
    {
        posImages = _posImages;
        negImages = _negImages;
        winSize = _winSize;

        posIdx = negIdx = 0;

        src.create( 0, 0 , CV_8UC1 );
        img.create( 0, 0, CV_8UC1 );
        point = offset = Point( 0, 0 );
        scale       = 1.0F;
        scaleFactor = 1.4142135623730950488016887242097F;
        stepFactor  = 0.5F;
        round = 0;

        indices = Common::RandSample(posImages.size(),posImages.size(),true);

        return true;
    }

    void restart() { posIdx = 0; }

    void nextNeg()
    {
        int count = negImages.size();
        for (int i = 0; i < count; i++) {
            src = negImages[negIdx++];

            round += negIdx / count;
            round = round % (winSize.width * winSize.height);
            negIdx %= count;

            offset.x = qMin( (int)round % winSize.width, src.cols - winSize.width );
            offset.y = qMin( (int)round / winSize.width, src.rows - winSize.height );
            if (!src.empty() && src.type() == CV_8UC1 && offset.x >= 0 && offset.y >= 0)
                break;
        }

        point = offset;
        scale = max(((float)winSize.width + point.x) / ((float)src.cols),
                    ((float)winSize.height + point.y) / ((float)src.rows));

        Size sz((int)(scale*src.cols + 0.5F), (int)(scale*src.rows + 0.5F));
        resize(src, img, sz);
    }

    bool getNeg(Mat &_img)
    {
        if (img.empty())
            nextNeg();

        Mat m(winSize.height, winSize.width, CV_8UC1, (void*)(img.data + point.y * img.step + point.x * img.elemSize()), img.step);
        m.copyTo(_img);

        if ((int)(point.x + (1.0F + stepFactor) * winSize.width) < img.cols)
            point.x += (int)(stepFactor * winSize.width);
        else {
            point.x = offset.x;
            if ((int)( point.y + (1.0F + stepFactor ) * winSize.height ) < img.rows)
                point.y += (int)(stepFactor * winSize.height);
            else {
                point.y = offset.y;
                scale *= scaleFactor;
                if (scale <= 1.0F)
                    resize(src, img, Size((int)(scale*src.cols), (int)(scale*src.rows)));
                else
                    nextNeg();
            }
        }
        return true;
    }

    bool getPos(Mat &_img)
    {
        if (posIdx >= posImages.size())
            return false;

        posImages[indices[posIdx++]].copyTo(_img);
        return true;
    }

    QList<Mat> posImages, negImages;

    int posIdx, negIdx;

    Mat     src, img;
    Point   offset, point;
    float   scale;
    float   scaleFactor;
    float   stepFactor;
    size_t  round;
    Size    winSize;

    QList<int> indices;
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

    BR_PROPERTY(QString, stageDescription, "")
    BR_PROPERTY(int, numStages, 20)
    BR_PROPERTY(int, numPos, 1000)
    BR_PROPERTY(int, numNegs, 1000)
    BR_PROPERTY(float, maxFAR, pow(0.5, numStages))

    QList<Classifier *> stages;

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        QList<Mat> posImages, negImages;
        for (int i = 0; i < images.size(); i++)
            labels[i] == 1 ? posImages.append(images[i]) : negImages.append(images[i]);

        stages.reserve(numStages);
        for (int i = 0; i < numStages; i++) {
            Classifier *next_stage = Classifier::make(stageDescription, NULL);
            stages.append(next_stage);
        }

        ImageHandler imgHandler;
        imgHandler.create(posImages, negImages, windowSize());

        for (int i = 0; i < numStages; i++) {
            qDebug() << "===== TRAINING" << i << "stage =====";
            qDebug() << "<BEGIN";

            QList<Mat> trainingImages;
            QList<float> trainingLabels;

            float currFAR = fillTrainingSet(imgHandler, trainingImages, trainingLabels);

            if (currFAR < maxFAR) {
                qDebug() << "FAR is below required level! Terminating early";
                return;
            }

            stages[i]->train(trainingImages, trainingLabels);

            qDebug() << "END>";
        }
    }

    float classify(const Mat &image, bool process, float *confidence) const
    {
        float stageConf = 0.0f;
        foreach (const Classifier *stage, stages) {
            float result = stage->classify(image, process, &stageConf);
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

    int maxCatCount() const
    {
        return stages.first()->maxCatCount();
    }

    Mat preprocess(const Mat &image) const
    {
        return stages.first()->preprocess(image);
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
    float fillTrainingSet(ImageHandler &imgHandler, QList<Mat> &images, QList<float> &labels)
    {
        imgHandler.restart();

        float confidence = 0.0f;

        while (images.size() < numPos) {
            Mat pos(imgHandler.winSize, CV_8UC1);
            if (!imgHandler.getPos(pos))
                qFatal("Cannot get another positive sample!");

            if (classify(pos, true, &confidence) > 0.0f) {
                printf("POS current samples: %d\r", images.size());
                images.append(pos);
                labels.append(1.0f);
            }
        }

        int posCount = images.size();
        qDebug() << "POS count : consumed  " << posCount << ":" << imgHandler.posIdx;

        int passedNegs = 0;
        while ((images.size() - posCount) < numNegs) {
            Mat neg(imgHandler.winSize, CV_8UC1);
            if (!imgHandler.getNeg(neg))
                qFatal("Cannot get another negative sample!");

            if (classify(neg, true, &confidence) > 0.0f) {
                printf("NEG current samples: %d\r", images.size() - posCount);
                images.append(neg);
                labels.append(0.0f);
            }
            passedNegs++;
        }

        double acceptanceRatio = (images.size() - posCount) / (double)passedNegs;
        qDebug() << "NEG count : acceptanceRatio  " << images.size() - posCount << ":" << acceptanceRatio;
        return acceptanceRatio;
    }
};

BR_REGISTER(Classifier, CascadeClassifier)

} // namespace br

#include "classification/cascade.moc"
