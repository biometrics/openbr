#ifndef ADABOOST_H
#define ADABOOST_H

#include "trainingdata.h"
#include <openbr/openbr_plugin.h>

namespace br
{

struct Node
{
    Node(int feature_idx, float threshold, int polarity, float weight) : feature_idx(feature_idx), threshold(threshold), polarity(polarity), weight(weight) {}
    int predict(float feature_response) const { return (polarity * (feature_response - threshold)) > 0 ? 1 : -1; }

    int feature_idx; // index of the feature corresponding to this weak classifier
    float threshold; // threshold at which classifier predicts yes or no
    int polarity; // 1 or -1. 1 if high responses correspond to positives, -1 if low values do.
    float weight; // weight of this classifier. Higher weight = more accurate classifier
};

struct DecisionTree
{
    DecisionTree() {}
    void append(Node node); // add a node to the tree
    float predict(QList<float> responses) const; // progress down the tree. Return value should be the
                                                 // sum of the weights of the correct nodes.

    QList<Node> tree; // List of the weak classifiers
};

class AdaBoost
{
public:
    AdaBoost();
    DecisionTree train(int num_classifiers); // mostly a wrapper to call selectBestNode() and updateWeights() num_classifiers times.

private:
    Node selectBestNode(); // search through all possible features to find one that improves performance the most
    void updateWeights(); // successful classification lowers weights, unsuccesful raises weights. After changes weights are normalized between 0 and 1

    TrainingData td; // training data to train on
    DecisionTree classifier; // classifier to be trained
};

}
#endif // ADABOOST_H
