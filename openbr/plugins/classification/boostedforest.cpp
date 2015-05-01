#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/boost.h>

using namespace cv;

namespace br
{

class BoostedForestClassifier : public Classifier
{
    Q_OBJECT

    Q_PROPERTY(br::Representation *representation READ get_representation WRITE set_representation RESET reset_representation STORED false)
    Q_PROPERTY(float minTAR READ get_minTAR WRITE set_minTAR RESET reset_minTAR STORED false)
    Q_PROPERTY(float maxFAR READ get_maxFAR WRITE set_maxFAR RESET reset_maxFAR STORED false)
    Q_PROPERTY(float trimRate READ get_trimRate WRITE set_trimRate RESET reset_trimRate STORED false)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED false)
    Q_PROPERTY(int maxWeakCount READ get_maxWeakCount WRITE set_maxWeakCount RESET reset_maxWeakCount STORED false)

    BR_PROPERTY(br::Representation *, representation, NULL)
    BR_PROPERTY(float, minTAR, 0.995)
    BR_PROPERTY(float, maxFAR, 0.5)
    BR_PROPERTY(float, trimRate, 0.95)
    BR_PROPERTY(int, maxDepth, 1)
    BR_PROPERTY(int, maxWeakCount, 100)

    CascadeBoost *boost;
    FeatureEvaluator *featureEvaluator;

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        CascadeBoostParams params(CvBoost::GENTLE, minTAR, maxFAR, trimRate, maxDepth, maxWeakCount);

        featureEvaluator = new FeatureEvaluator;
        featureEvaluator->init(representation, images.size());

        for (int i = 0; i < images.size(); i++)
            featureEvaluator->setImage(images[i], labels[i], i);

        boost = new CascadeBoost;
        boost->train(featureEvaluator, images.size(), 1024, 1024, params);
    }

    float classify(const Mat &image) const
    {
        featureEvaluator->setImage(image, 0, 0);
        return boost->predict(0);
    }

    int numFeatures() const
    {
        return representation->numFeatures();
    }

    int maxCatCount() const
    {
        return representation->maxCatCount();
    }

    Size windowSize() const
    {
        return representation->preWindowSize();
    }

    void getUsedFeatures(Mat &featureMap) const
    {
        boost->markUsedFeaturesInMap(featureMap);
    }

    void write(FileStorage &fs, const Mat &featureMap) const
    {
        boost->write(fs, featureMap);
    }

    void writeFeatures(FileStorage &fs, const Mat &featureMap) const
    {
        featureEvaluator->writeFeatures(fs, featureMap);
    }
};

BR_REGISTER(Classifier, BoostedForestClassifier)

} // namespace br

#include "classification/boostedforest.moc"
