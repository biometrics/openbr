#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/boost.h>

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

        posImages[posIdx++].copyTo(_img);
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
};

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

        ImageHandler imgHandler;
        imgHandler.create(posImages, negImages, Size(24, 24));

        stages.reserve(numStages);
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

            Classifier *next_stage = Classifier::make(stageDescription, NULL);
            next_stage->train(trainingImages, trainingLabels);
            stages.append(next_stage);

            qDebug() << "END>";
        }
    }

    float classify(const Mat &image) const
    {
        foreach (const Classifier *stage, stages)
            if (stage->classify(image) == 0.0f)
                return 0.0f;
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

    cv::Size windowSize() const
    {
        return stages.first()->windowSize();
    }

    void write(FileStorage &fs) const
    {
        fs << "stageCount" << stages.size();

        fs << "stages" << "[";
        foreach (const Classifier *stage, stages) {
            fs << "{";
            stage->write(fs);
            fs << "}";
        }
        fs << "]";
    }

private:
    float fillTrainingSet(ImageHandler &imgHandler, QList<Mat> &images, QList<float> &labels)
    {
        imgHandler.restart();

        while (images.size() < numPos) {
            Mat pos(imgHandler.winSize, CV_8UC1);
            if (!imgHandler.getPos(pos))
                qFatal("Cannot get another positive sample!");

            if (classify(pos) == 1.0f) {
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

            if (classify(neg) == 1.0f) {
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
