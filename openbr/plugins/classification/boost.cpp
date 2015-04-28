#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/boost.h>
#include <openbr/core/features.h>

using namespace cv;

namespace br
{

class BoostClassifier : public Classifier
{
    Q_OBJECT

    Q_PROPERTY(float minTAR READ get_minTAR WRITE set_minTAR RESET reset_minTAR STORED false)
    Q_PROPERTY(float maxFAR READ get_maxFAR WRITE set_maxFAR RESET reset_maxFAR STORED false)
    Q_PROPERTY(float trimRate READ get_trimRate WRITE set_trimRate RESET reset_trimRate STORED false)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED false)
    Q_PROPERTY(int maxWeakCount READ get_maxWeakCount WRITE set_maxWeakCount RESET reset_maxWeakCount STORED false)

    BR_PROPERTY(float, minTAR, 0.995)
    BR_PROPERTY(float, maxFAR, 0.5)
    BR_PROPERTY(float, trimRate, 0.95)
    BR_PROPERTY(int, maxDepth, 1)
    BR_PROPERTY(int, maxWeakCount, 100)

    CascadeBoost *boost;

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        FeatureEvaluator *featureEvaluator = FeatureEvaluator::create(0);
        featureEvaluator->init(new FeatureParams, images.size(), Size(24, 24));
        for (int i = 0; i < images.size(); i++)
            featureEvaluator->setImage(images[i], (uchar)labels[i], i);

        CascadeBoostParams stageParams(CvBoost::GENTLE, minTAR, maxFAR, trimRate, maxDepth, maxWeakCount);

        if (boost) delete boost;

        boost = new CascadeBoost;
        boost->train(featureEvaluator, images.size(), 1024, 1024, stageParams);
    }

    float classify(const Mat &image) const
    {
        (void) image;
        return 0.;
    }
};

BR_REGISTER(Classifier, BoostClassifier)

} // namespace br

#include "classification/boost.moc"
