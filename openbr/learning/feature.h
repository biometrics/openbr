#ifndef FEATURE_H
#define FEATURE_H

#include <openbr/openbr_plugin.h>

// These two classes should be subclassed to handle any type of feature. In a perfect world I would like to subclass
// only Feature, and have FeatureEvaluator just call randomize and evaluate (and thus not have to be subclassed) as
// that would be simpler.
namespace br
{

class Feature
{
public:
    Feature();
    QList<int> get_definition() { return definition; }

    virtual void randomize(int height, int width, int channels) = 0; // Initialize the feature within these dimensions
    virtual float evaluate(const QList<cv::Mat> &img, const cv::Size &pos) const = 0; // img is a list of mats to support multiple channels
                                                                                      // if the image is larger than the feature model (i.e
                                                                                      // a real image instead of a training image) pos determines
                                                                                      // the top left corner of the feature.

protected:
    QList<int> definition; // The definition of the feature (usually these are coords)
};

class FeatureEvaluator
{
public:
    FeatureEvaluator(int num_features_) : num_features(num_features_) {}
    int get_num_features() const { return num_features; }
    QList<Feature> get_features() const { return features; }

    virtual void generateRandomFeatures(int height, int width, int channels) = 0; // initialize num_features random features within these dimensions.
                                                                                  // will call Feature->randomize()
    virtual QList<float> evaluate(const QList<cv::Mat> &img, cv::Size pos) const = 0; // return the response of all features on the given image
                                                                                      // will call Feature->evaluate()

private:
    int num_features;
    QList<Feature> features;
};

}

#endif // FEATURE_H
