#ifndef FEATURE_H
#define FEATURE_H

#include <openbr/openbr_plugin.h>

// These two classes should be subclassed to handle any type of feature. In a perfect world I would like to subclass
// only Feature, and have FeatureEvaluator just call randomize and evaluate (and thus not have to be subclassed) as
// that would be simpler.
namespace br
{

class Feature : public Object
{
public:
    Feature() {}
    QList<int> get_definition() { return definition; }

    virtual void randomize(int height, int width, int channels) { return; } // Initialize the feature within these dimensions
    virtual float evaluate(const cv::Mat &img) const { return 0.; } // returns the feature response for a given image

    void load(QDataStream &stream) { stream >> definition; }
    void store(QDataStream &stream) const { stream << definition; }

protected:
    QList<int> definition; // The definition of the feature (usually these are coords)
};

class FeatureEvaluator
{
public:
    FeatureEvaluator() {}
    FeatureEvaluator(QList<Feature> features) : features(features) {}

    int num_features() const { return features.size(); }
    QList<Feature> get_features() const { return features; }

    virtual void generateRandomFeatures(int num_features, int height, int width, int channels) { return; } // initialize num_features random features within these dimensions.

    virtual float evaluate(const cv::Mat &img, const int feature_idx) { return features[feature_idx].evaluate(img); }
    virtual QList<float> evaluateAll(const cv::Mat &img) const { return QList<float>(); } // return the response of all features on the given image

private:
    QList<Feature> features;
};

}

#endif // FEATURE_H
