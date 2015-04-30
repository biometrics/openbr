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

#define CV_SUM_OFFSETS( p0, p1, p2, p3, rect, step )                      \
    /* (x, y) */                                                          \
    (p0) = (rect).x + (step) * (rect).y;                                  \
    /* (x + w, y) */                                                      \
    (p1) = (rect).x + (rect).width + (step) * (rect).y;                   \
    /* (x + w, y) */                                                      \
    (p2) = (rect).x + (step) * ((rect).y + (rect).height);                \
    /* (x + w, y + h) */                                                  \
    (p3) = (rect).x + (rect).width + (step) * ((rect).y + (rect).height);

namespace br
{

template<class Feature>
void _writeFeatures( const std::vector<Feature> features, cv::FileStorage &fs, const cv::Mat& featureMap )
{
    fs << CC_FEATURES << "[";
    const cv::Mat_<int>& featureMap_ = (const cv::Mat_<int>&)featureMap;
    for ( int fi = 0; fi < featureMap.cols; fi++ )
        if ( featureMap_(0, fi) >= 0 )
        {
            fs << "{";
            features[fi].write( fs );
            fs << "}";
        }
    fs << "]";
}

class FeatureEvaluator
{
public:
    ~FeatureEvaluator() {}
    void init(int _maxSampleCount, cv::Size _winSize);
    void setImage(const cv::Mat& img, uchar clsLabel, int idx);
    void writeFeatures(cv::FileStorage &fs, const cv::Mat& featureMap) const;
    float operator()(int featureIdx, int sampleIdx) const { return (float)features[featureIdx].calc(data, sampleIdx); }

    int getNumFeatures() const { return numFeatures; }
    int getMaxCatCount() const { return maxCatCount; }
    int getFeatureSize() const { return featSize; }
    const cv::Mat& getCls() const { return cls; }
    float getCls(int si) const { return cls.at<float>(si, 0); }

protected:
    void generateFeatures();

    class Feature
    {
    public:
        Feature();
        Feature( int offset, int x, int y, int _block_w, int _block_h  );
        uchar calc( const cv::Mat &data, int y ) const;
        void write( cv::FileStorage &fs ) const;

        cv::Rect rect;
        int p[16];
    };
    std::vector<Feature> features;

    cv::Mat data, cls;

    int npos, nneg;
    int numFeatures;
    int maxCatCount; // 0 in case of numerical features
    int featSize; // 1 in case of simple features (HAAR, LBP) and N_BINS(9)*N_CELLS(4) in case of Dalal's HOG features

    cv::Size winSize;
};

inline uchar FeatureEvaluator::Feature::calc(const cv::Mat &data, int y) const
{
    const int* ptr = data.ptr<int>(y);
    int cval = ptr[p[5]] - ptr[p[6]] - ptr[p[9]] + ptr[p[10]];

    return (uchar)((ptr[p[0]] - ptr[p[1]] - ptr[p[4]] + ptr[p[5]] >= cval ? 128 : 0) |   // 0
        (ptr[p[1]] - ptr[p[2]] - ptr[p[5]] + ptr[p[6]] >= cval ? 64 : 0) |    // 1
        (ptr[p[2]] - ptr[p[3]] - ptr[p[6]] + ptr[p[7]] >= cval ? 32 : 0) |    // 2
        (ptr[p[6]] - ptr[p[7]] - ptr[p[10]] + ptr[p[11]] >= cval ? 16 : 0) |  // 5
        (ptr[p[10]] - ptr[p[11]] - ptr[p[14]] + ptr[p[15]] >= cval ? 8 : 0) | // 8
        (ptr[p[9]] - ptr[p[10]] - ptr[p[13]] + ptr[p[14]] >= cval ? 4 : 0) |  // 7
        (ptr[p[8]] - ptr[p[9]] - ptr[p[12]] + ptr[p[13]] >= cval ? 2 : 0) |   // 6
        (ptr[p[4]] - ptr[p[5]] - ptr[p[8]] + ptr[p[9]] >= cval ? 1 : 0));     // 3
}

} // namespace br

#endif // FEATURE_H
