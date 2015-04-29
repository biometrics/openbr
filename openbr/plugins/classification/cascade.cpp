#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/cascade.h>

using namespace cv;

namespace br
{

class _CascadeClassifier : public Classifier
{
    Q_OBJECT

    Q_PROPERTY(br::Classifier *stage READ get_stage WRITE set_stage RESET reset_stage STORED false)
    Q_PROPERTY(int numStages READ get_numStages WRITE set_numStages RESET reset_numStages STORED false)
    Q_PROPERTY(int numNegs READ get_numNegs WRITE set_numNegs RESET reset_numNegs STORED false)
    Q_PROPERTY(float maxFAR READ get_maxFAR WRITE set_maxFAR RESET reset_maxFAR STORED false)

    BR_PROPERTY(br::Classifier *, stage, NULL)
    BR_PROPERTY(int, numStages, 20)
    BR_PROPERTY(int, numNegs, 1000)
    BR_PROPERTY(float, maxFAR, pow(0.5, numStages))

    QList<Classifier *> stages;

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        QList<Mat> posImages, negImages;
        for (int i = 0; i < images.size(); i++)
            labels[i] == 1 ? posImages.append(images[i]) : negImages.append(images[i]);

        QList<Mat> trainingImages;
        QList<float> trainingLabels;

        for (int i = 0; i < numStages; i++) {
            float currFAR = updateTrainingSet(posImages, negImages, trainingImages, trainingLabels);

            if (currFAR < maxFAR) {
                qDebug() << "FAR is below required level! Terminating early";
                return;
            }

            Classifier *next_stage = stage->clone();
            next_stage->train(trainingImages, trainingLabels);
            stages.append(next_stage);
        }
    }

    float classify(const Mat &image) const
    {
        (void) image;
        return 0.;
    }

    float updateTrainingSet(const QList<Mat> &posImages, const QList<Mat> &negImages, QList<Mat> &trainingImages, QList<float> &trainingLabels)
    {
        trainingImages.clear();
        trainingLabels.clear();

        foreach (const Mat &pos, posImages) {
            if (classify(pos) > 0) {
                trainingImages.append(pos);
                trainingLabels.append(1.);
            }
        }

        NegFinder finder(negImages, Size(24, 24));
        int totalNegs = 0, passedNegs = 0;
        while (true) {
            totalNegs++;
            Mat neg = finder.get();
            if (classify(neg) > 0) {
                trainingImages.append(neg);
                trainingLabels.append(0.);
                passedNegs++;
            }

            if (passedNegs >= numNegs)
                return passedNegs / (float)totalNegs;
        }
    }

private:
    struct NegFinder
    {
        NegFinder(const QList<Mat> &_negs, Size _winSize)
        {
            negs = _negs;
            winSize = _winSize;

            negIdx = round = 0;
            img.create( 0, 0, CV_8UC1 );
            point = offset = Point( 0, 0 );
            scale       = 1.0F;
            scaleFactor = 1.4142135623730950488016887242097F;
            stepFactor  = 0.5F;
        }

        void _next()
        {
            src = negs[negIdx++];

            round += negIdx / negs.size();
            round %= (winSize.width * winSize.height);
            negIdx %= negs.size();

            point = offset = Point(std::min(round % winSize.width, src.cols - winSize.width),
                                   std::min(round / winSize.width, src.rows - winSize.height));

            scale = max(((float)winSize.width + point.x) / ((float)src.cols),
                        ((float)winSize.height + point.y) / ((float)src.rows));

            Size sz((int)(scale*src.cols + 0.5F), (int)(scale*src.rows + 0.5F));
            resize(src, img, sz);
        }

        Mat get()
        {
            if (img.empty())
                _next();

            Mat neg(winSize, CV_8UC1);
            Mat m(winSize.height, winSize.width, CV_8UC1, (void*)(img.data + point.y * img.step + point.x * img.elemSize()), img.step);
            m.copyTo(neg);

            if ((int)(point.x + (1.0F + stepFactor ) * winSize.width) < img.cols)
                point.x += (int)(stepFactor * winSize.width);
            else {
                point.x = offset.x;
                if ((int)( point.y + (1.0F + stepFactor ) * winSize.height ) < img.rows)
                    point.y += (int)(stepFactor * winSize.height);
                else {
                    point.y = offset.y;
                    scale *= scaleFactor;
                    if( scale <= 1.0F )
                        resize(src, img, Size( (int)(scale*src.cols), (int)(scale*src.rows)));
                    else
                        _next();
                }
            }
            return neg;
        }

        QList<Mat> negs;
        int negIdx, round;
        Mat src, img;
        float scale;
        float scaleFactor;
        float stepFactor;
        Size winSize;
        Point offset, point;
    };
};

BR_REGISTER(Classifier, _CascadeClassifier)

} // namespace br

#include "classification/cascade.moc"
