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
#define CC_MAX_DEPTH        "maxDepth"
#define CC_WEAK_COUNT       "maxWeakCount"
#define CC_STAGE_THRESHOLD  "stageThreshold"
#define CC_WEAK_CLASSIFIERS "weakClassifiers"
#define CC_INTERNAL_NODES   "internalNodes"
#define CC_LEAF_VALUES      "leafValues"

#define CC_FEATURES       "features"
#define CC_FEATURE_PARAMS "featureParams"
#define CC_MAX_CAT_COUNT  "maxCatCount"
#define CC_FEATURE_SIZE   "featSize"

#define CC_HAAR        "HAAR"
#define CC_MODE        "mode"
#define CC_MODE_BASIC  "BASIC"
#define CC_MODE_CORE   "CORE"
#define CC_MODE_ALL    "ALL"
#define CC_RECTS       "rects"
#define CC_TILTED      "tilted"

#define CC_LBP  "LBP"
#define CC_RECT "rect"

#define CC_HOG "HOG"
#define CC_HOGMULTI "HOGMulti"

#define CC_NPD "NPD"
#define CC_POINTS "points"
#define CC_POINT "point"

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

#define CV_TILTED_OFFSETS( p0, p1, p2, p3, rect, step )                   \
    /* (x, y) */                                                          \
    (p0) = (rect).x + (step) * (rect).y;                                  \
    /* (x - h, y + h) */                                                  \
    (p1) = (rect).x - (rect).height + (step) * ((rect).y + (rect).height);\
    /* (x + w, y + w) */                                                  \
    (p2) = (rect).x + (rect).width + (step) * ((rect).y + (rect).width);  \
    /* (x + w - h, y + w + h) */                                          \
    (p3) = (rect).x + (rect).width - (rect).height                        \
           + (step) * ((rect).y + (rect).width + (rect).height);

namespace br
{

float calcNormFactor( const cv::Mat& sum, const cv::Mat& sqSum );

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

class Params
{
public:
    Params();
    virtual ~Params() {}
    // from|to file
    virtual void write( cv::FileStorage &fs ) const = 0;
    virtual bool read( const cv::FileNode &node ) = 0;
    // from|to screen
    virtual void printDefaults() const;
    virtual void printAttrs() const;
    virtual bool scanAttr( const std::string prmName, const std::string val );
    std::string name;
};

class FeatureParams : public Params
{
public:
    enum { LBP = 0 };
    FeatureParams();
    virtual void init( const FeatureParams& fp );
    virtual void write( cv::FileStorage &fs ) const;
    virtual bool read( const cv::FileNode &node );
    static cv::Ptr<FeatureParams> create( int featureType );
    int maxCatCount; // 0 in case of numerical features
    int featSize; // 1 in case of simple features (HAAR, LBP) and N_BINS(9)*N_CELLS(4) in case of Dalal's HOG features
};

class FeatureEvaluator
{
public:
    virtual ~FeatureEvaluator() {}
    virtual void init(const FeatureParams *_featureParams,
                      int _maxSampleCount, cv::Size _winSize );
    virtual void setImage(const cv::Mat& img, uchar clsLabel, int idx);
    virtual void writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const = 0;
    virtual float operator()(int featureIdx, int sampleIdx) const = 0;
    static cv::Ptr<FeatureEvaluator> create(int type);

    int getNumFeatures() const { return numFeatures; }
    int getMaxCatCount() const { return featureParams->maxCatCount; }
    int getFeatureSize() const { return featureParams->featSize; }
    const cv::Mat& getCls() const { return cls; }
    float getCls(int si) const { return cls.at<float>(si, 0); }
protected:
    virtual void generateFeatures() = 0;

    int npos, nneg;
    int numFeatures;
    cv::Size winSize;
    FeatureParams *featureParams;
    cv::Mat cls;
};


//------------------------- LBP Feature ---------------------------------

#define LBPF_NAME "lbpFeatureParams"

struct LBPFeatureParams : FeatureParams
{
    LBPFeatureParams();

};

class LBPEvaluator : public FeatureEvaluator
{
public:
    virtual ~LBPEvaluator() {}
    virtual void init(const FeatureParams *_featureParams,
        int _maxSampleCount, cv::Size _winSize );
    virtual void setImage(const cv::Mat& img, uchar clsLabel, int idx);
    virtual float operator()(int featureIdx, int sampleIdx) const
    { return (float)features[featureIdx].calc( sum, sampleIdx); }
    virtual void writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const;
protected:
    virtual void generateFeatures();

    class Feature
    {
    public:
        Feature();
        Feature( int offset, int x, int y, int _block_w, int _block_h  );
        uchar calc( const cv::Mat& _sum, size_t y ) const;
        void write( cv::FileStorage &fs ) const;

        cv::Rect rect;
        int p[16];
    };
    std::vector<Feature> features;

    cv::Mat sum;
};

inline uchar LBPEvaluator::Feature::calc(const cv::Mat &_sum, size_t y) const
{
    const int* psum = _sum.ptr<int>((int)y);
    int cval = psum[p[5]] - psum[p[6]] - psum[p[9]] + psum[p[10]];

    return (uchar)((psum[p[0]] - psum[p[1]] - psum[p[4]] + psum[p[5]] >= cval ? 128 : 0) |   // 0
        (psum[p[1]] - psum[p[2]] - psum[p[5]] + psum[p[6]] >= cval ? 64 : 0) |    // 1
        (psum[p[2]] - psum[p[3]] - psum[p[6]] + psum[p[7]] >= cval ? 32 : 0) |    // 2
        (psum[p[6]] - psum[p[7]] - psum[p[10]] + psum[p[11]] >= cval ? 16 : 0) |  // 5
        (psum[p[10]] - psum[p[11]] - psum[p[14]] + psum[p[15]] >= cval ? 8 : 0) | // 8
        (psum[p[9]] - psum[p[10]] - psum[p[13]] + psum[p[14]] >= cval ? 4 : 0) |  // 7
        (psum[p[8]] - psum[p[9]] - psum[p[12]] + psum[p[13]] >= cval ? 2 : 0) |   // 6
        (psum[p[4]] - psum[p[5]] - psum[p[8]] + psum[p[9]] >= cval ? 1 : 0));     // 3
}

} // namespace br

#endif // FEATURE_H
