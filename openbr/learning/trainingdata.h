#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include "feature.h"
#include <openbr/openbr_plugin.h>

namespace br
{

struct FeatureResponse // Feature vectors need to be sorted to do boosting. This keeps the weights and labels properly associated with the response without headaches.
{
    FeatureResponse(float response, int label, float weight) : response(response), label(label), weight(weight) {}

    float response;
    int label;
    float weight;
};

typedef QList<FeatureResponse> FeatureVector;

class TrainingData
{
public:
    TrainingData();
    // This can have getters and setters to make manipulating feature vectors easier

private:
    QList<FeatureVector> data;
    QList<bool> visited_features;
};

}
#endif // TRAININGDATA_H
