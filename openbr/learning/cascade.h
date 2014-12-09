#ifndef CASCADE_H
#define CASCADE_H

#include "feature.h"
#include "adaboost.h"
#include <openbr/openbr_plugin.h>

namespace br
{

// I'm not sure if this class can be made general enough (i.e not just for face detection) to live here.
// Ideally, it should handle model i/o, and probably detection as well. Perhaps it is better in plugins/cascade.cpp
class Cascade
{
public:
    Cascade(const DecisionTree &classifier, const FeatureEvaluator &evaluator) : classifier(classifier), evaluator(evaluator) {}
    void detect(const cv::Mat &img, QList<QRectF> &detections, QList<float> &confidences);

    void write(QString path);
    static Cascade read(QString path);

private:
    DecisionTree classifier;
    FeatureEvaluator evaluator;
};

}

#endif // CASCADE_H
