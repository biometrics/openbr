#ifndef _BOOST_H_
#define _BOOST_H_

#include "ml.h"
#include <openbr/openbr_plugin.h>

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

struct FeatureEvaluator
{
    ~FeatureEvaluator() {}
    void init(Representation *_representation, int _maxSampleCount);
    void setImage(const cv::Mat& img, uchar clsLabel, int idx);
    float operator()(int featureIdx, int sampleIdx) const { return representation->evaluate(data.row(sampleIdx), featureIdx); }

    int getNumFeatures() const { return representation->numFeatures(); }
    int getMaxCatCount() const { return representation->maxCatCount(); }
    int getFeatureSize() const { return 1; }
    const cv::Mat& getCls() const { return cls; }
    float getCls(int si) const { return cls.at<float>(si, 0); }

    cv::Mat data, cls;
    Representation *representation;
};

struct CascadeBoostParams : CvBoostParams
{
    float minHitRate;
    float maxFalseAlarm;

    CascadeBoostParams();
    CascadeBoostParams(int _boostType, float _minHitRate, float _maxFalseAlarm,
                       double _weightTrimRate, int _maxDepth, int _maxWeakCount);
    virtual ~CascadeBoostParams() {}
};

struct CascadeBoostTrainData : CvDTreeTrainData
{
    CascadeBoostTrainData(const FeatureEvaluator* _featureEvaluator, const CvDTreeParams& _params);
    CascadeBoostTrainData(const FeatureEvaluator* _featureEvaluator,
                          int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize,
                          const CvDTreeParams& _params = CvDTreeParams());
    virtual void setData(const FeatureEvaluator* _featureEvaluator,
                         int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize,
                         const CvDTreeParams& _params=CvDTreeParams());
    void precalculate();

    virtual CvDTreeNode* subsample_data(const CvMat* _subsample_idx);

    virtual const int* get_class_labels(CvDTreeNode* n, int* labelsBuf);
    virtual const int* get_cv_labels(CvDTreeNode* n, int* labelsBuf);
    virtual const int* get_sample_indices(CvDTreeNode* n, int* indicesBuf);

    virtual void get_ord_var_data(CvDTreeNode* n, int vi, float* ordValuesBuf, int* sortedIndicesBuf,
                                  const float** ordValues, const int** sortedIndices, int* sampleIndicesBuf);
    virtual const int* get_cat_var_data(CvDTreeNode* n, int vi, int* catValuesBuf);
    virtual float getVarValue(int vi, int si);
    virtual void free_train_data();

    const FeatureEvaluator* featureEvaluator;
    cv::Mat valCache; // precalculated feature values (CV_32FC1)
    CvMat _resp; // for casting
    int numPrecalcVal, numPrecalcIdx;
};

class CascadeBoostTree : public CvBoostTree
{
public:
    virtual CvDTreeNode* predict(int sampleIdx) const;

protected:
    virtual void split_node_data(CvDTreeNode* n);
};

class CascadeBoost : public CvBoost
{
public:
    virtual bool train(const FeatureEvaluator *_featureEvaluator,
                       int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize,
                       const CascadeBoostParams &_params=CascadeBoostParams());
    virtual float predict( int sampleIdx, bool returnSum = false ) const;

    float getThreshold() const { return threshold; }
<<<<<<< HEAD
    QList<CvBoostTree*> getClassifiers() const { return classifiers; }
=======
    const QList<CvBoostTree*> getTrees() const { return trees; }
>>>>>>> 4fab7f69ddc82d6ba40a73fc6233e3cc9871473e

protected:
    virtual bool set_params(const CvBoostParams& _params);
    virtual void update_weights(CvBoostTree* tree);
    virtual bool isErrDesired();

<<<<<<< HEAD
    QList<CvBoostTree*> classifiers;
=======
    QList<CvBoostTree*> trees;
>>>>>>> 4fab7f69ddc82d6ba40a73fc6233e3cc9871473e

    float threshold;
    float minHitRate, maxFalseAlarm;
};

} // namespace br

#endif

