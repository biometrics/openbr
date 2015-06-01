#ifndef _BOOST_H_
#define _BOOST_H_

#include "ml.h"
#include <openbr/openbr_plugin.h>

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
    void init(Representation *_representation, int _maxSampleCount, int channels);
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
    CascadeBoostTrainData(const FeatureEvaluator* _featureEvaluator, int _channels, const CvDTreeParams& _params);
    CascadeBoostTrainData(const FeatureEvaluator* _featureEvaluator,
                          int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize, int _channels,
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
    int numPrecalcVal, numPrecalcIdx, channels;
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
    virtual void train(const FeatureEvaluator *_featureEvaluator,
                       int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize, int _channels,
                       const CascadeBoostParams &_params=CascadeBoostParams());
    virtual float predict( int sampleIdx, bool returnSum = false ) const;

    float getThreshold() const { return threshold; }
    QList<CvBoostTree *> getClassifers() const { return classifiers; }

protected:
    virtual bool set_params(const CvBoostParams& _params);
    virtual void update_weights(CvBoostTree* tree);
    virtual bool isErrDesired();

    QList<CvBoostTree *> classifiers;

    float threshold;
    float minHitRate, maxFalseAlarm;
    int channels;
};

} // namespace br

#endif

