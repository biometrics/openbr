#ifndef FEATURE_H
#define FEATURE_H

#include <openbr/openbr_plugin.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

#define CC_CASCADE_FILENAME "cascade.xml"
#define CC_PARAMS_FILENAME "params.xml"

#define CC_CASCADE_PARAMS "cascadeParams"
#define CC_STAGE_TYPE "stageType"
#define CC_FEATURE_TYPE "featureType"
#define CC_HEIGHT "height"
#define CC_WIDTH  "width"

#define CC_STAGE_NUM    "stageNum"
#define CC_STAGES       "stages"
#define CC_STAGE_PARAMS "stageParams"

#define CC_BOOST            "BOOST"
#define CC_BOOST_TYPE       "boostType"
#define CC_DISCRETE_BOOST   "DAB"
#define CC_REAL_BOOST       "RAB"
#define CC_LOGIT_BOOST      "LB"
#define CC_GENTLE_BOOST     "GAB"
#define CC_MINHITRATE       "minHitRate"
#define CC_MAXFALSEALARM    "maxFalseAlarm"
#define CC_TRIM_RATE        "weightTrimRate"
#define CC_MAX_DEPTH        "maxDeptrh"
#define CC_WEAK_COUNT       "maxWeakCount"
#define CC_STAGE_THRESHOLD  "stageThreshold"
#define CC_WEAK_CLASSIFIERS "weakClassifiers"
#define CC_INTERNAL_NODES   "internalNodes"
#define CC_LEAF_VALUES      "leafValues"

#define CC_FEATURES       "features"
#define CC_FEATURE_PARAMS "featureParams"
#define CC_MAX_CAT_COUNT  "maxCatCount"
#define CC_FEATURE_SIZE   "featSize"

#define CC_LBP  "LBP"
#define CC_RECT "rect"

#ifdef _WIN32
#define TIME( arg ) (((double) clock()) / CLOCKS_PER_SEC)
#else
#define TIME( arg ) (time( arg ))
#endif

namespace br
{

class FeatureEvaluator
{
public:
    ~FeatureEvaluator() {}
    void init(Representation *_representation, int _maxSampleCount);
    void setImage(const cv::Mat& img, uchar clsLabel, int idx);
    void writeFeatures(cv::FileStorage &fs, const cv::Mat& featureMap) const;
    float operator()(int featureIdx, int sampleIdx) const { return representation->evaluate(data.row(sampleIdx), featureIdx); }

    int getNumFeatures() const { return representation->numFeatures(); }
    int getMaxCatCount() const { return representation->maxCatCount(); }
    int getFeatureSize() const { return 1; }
    const cv::Mat& getCls() const { return cls; }
    float getCls(int si) const { return cls.at<float>(si, 0); }

    cv::Mat data, cls;
    Representation *representation;
};

} // namespace br

#endif // FEATURE_H
