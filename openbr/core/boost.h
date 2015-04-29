#ifndef _BOOST_H_
#define _BOOST_H_

#include "features.h"
#include "ml.h"

namespace br
{

struct CascadeBoostParams : CvBoostParams
{
    float minHitRate;
    float maxFalseAlarm;

    CascadeBoostParams();
    CascadeBoostParams( int _boostType, float _minHitRate, float _maxFalseAlarm,
                          double _weightTrimRate, int _maxDepth, int _maxWeakCount );
    virtual ~CascadeBoostParams() {}
    void write( cv::FileStorage &fs ) const;
};

struct BoostBRCompatibility
{
    BoostBRCompatibility(Representation *_representation, int _numSamples);
    void setImage(const cv::Mat &img, uchar clsLabel, int idx);
    float operator()(int featureIdx, int sampleIdx) const;
    const cv::Mat &getCls() const { return cls; }
    float getCls(int idx) const { return cls.at<float>(idx, 0); }

    int numFeatures() const { return representation->numFeatures(); }
    int maxCatCount() const { return 256; }

    Representation *representation;
    cv::Mat data, cls;
};

struct CascadeBoostTrainData : CvDTreeTrainData
{
    CascadeBoostTrainData( const BoostBRCompatibility* _compat,
                             const CvDTreeParams& _params );
    CascadeBoostTrainData( const BoostBRCompatibility* _compat,
                             int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize,
                             const CvDTreeParams& _params = CvDTreeParams() );
    virtual void setData( const BoostBRCompatibility* _compat,
                          int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize,
                          const CvDTreeParams& _params=CvDTreeParams() );
    void precalculate();

    virtual CvDTreeNode* subsample_data( const CvMat* _subsample_idx );

    virtual const int* get_class_labels( CvDTreeNode* n, int* labelsBuf );
    virtual const int* get_cv_labels( CvDTreeNode* n, int* labelsBuf);
    virtual const int* get_sample_indices( CvDTreeNode* n, int* indicesBuf );

    virtual void get_ord_var_data( CvDTreeNode* n, int vi, float* ordValuesBuf, int* sortedIndicesBuf,
                                  const float** ordValues, const int** sortedIndices, int* sampleIndicesBuf );
    virtual const int* get_cat_var_data( CvDTreeNode* n, int vi, int* catValuesBuf );
    virtual float getVarValue( int vi, int si );
    virtual void free_train_data();

    const BoostBRCompatibility* compat;
    cv::Mat valCache; // precalculated feature values (CV_32FC1)
    CvMat _resp; // for casting
    int numPrecalcVal, numPrecalcIdx;
};

class CascadeBoostTree : public CvBoostTree
{
public:
    virtual CvDTreeNode* predict( int sampleIdx ) const;
    void write( cv::FileStorage &fs, const cv::Mat& featureMap );
    void markFeaturesInMap( cv::Mat& featureMap );
protected:
    virtual void split_node_data( CvDTreeNode* n );
};

class CascadeBoost : public CvBoost
{
public:
    virtual bool train( const BoostBRCompatibility* _compat,
                        int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize,
                        const CascadeBoostParams& _params=CascadeBoostParams() );
    virtual float predict( int sampleIdx, bool returnSum = false ) const;

    float getThreshold() const { return threshold; }
    void write( cv::FileStorage &fs, const cv::Mat& featureMap ) const;
    void markUsedFeaturesInMap( cv::Mat& featureMap );
protected:
    virtual bool set_params( const CvBoostParams& _params );
    virtual void update_weights( CvBoostTree* tree );
    virtual bool isErrDesired();

    float threshold;
    float minHitRate, maxFalseAlarm;
};

} // namespace br

#endif

