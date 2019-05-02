#include <opencv2/imgproc/imgproc.hpp>
#include <openbr/core/opencvutils.h>
#include <QDebug>

#include "boost.h"
#include "cxmisc.h"

using namespace std;
using namespace br;
using namespace cv;

static inline double
logRatio( double val )
{
    const double eps = 1e-5;

    val = max( val, eps );
    val = min( val, 1. - eps );
    return log( val/(1. - val) );
}

#define CV_CMP_NUM_IDX(i,j) (aux[i] < aux[j])
static CV_IMPLEMENT_QSORT_EX( icvSortIntAux, int, CV_CMP_NUM_IDX, const float* )

#define CV_THRESHOLD_EPS (0.00001F)

static const int MinBlockSize = 1 << 16;
static const int BlockSizeDelta = 1 << 10;

// TODO remove this code duplication with ml/precomp.hpp

static int CV_CDECL icvCmpIntegers( const void* a, const void* b )
{
    return *(const int*)a - *(const int*)b;
}

static CvMat* cvPreprocessIndexArray( const CvMat* idx_arr, int data_arr_size, bool check_for_duplicates=false, const int channels = 1 )
{
    CvMat* idx = 0;

    CV_FUNCNAME( "cvPreprocessIndexArray" );

    __BEGIN__;

    int i, idx_total, idx_selected = 0, step, type, prev = INT_MIN, is_sorted = 1;
    uchar* srcb = 0;
    int* srci = 0;
    int* dsti;

    if( !CV_IS_MAT(idx_arr) )
        CV_ERROR( CV_StsBadArg, "Invalid index array" );

    if( idx_arr->rows != 1 && idx_arr->cols != 1 )
        CV_ERROR( CV_StsBadSize, "the index array must be 1-dimensional" );

    idx_total = idx_arr->rows + idx_arr->cols - 1;
    srcb = idx_arr->data.ptr;
    srci = idx_arr->data.i;

    type = CV_MAT_TYPE(idx_arr->type);
    step = CV_IS_MAT_CONT(idx_arr->type) ? 1 : idx_arr->step/CV_ELEM_SIZE(type);

    if (type == CV_8UC(channels) || type == CV_8SC1) {
        // idx_arr is array of 1's and 0's -
        // i.e. it is a mask of the selected components
        if( idx_total != data_arr_size )
            CV_ERROR( CV_StsUnmatchedSizes,
            "Component mask should contain as many elements as the total number of input variables" );

        for( i = 0; i < idx_total; i++ )
            idx_selected += srcb[i*step] != 0;

        if( idx_selected == 0 )
            CV_ERROR( CV_StsOutOfRange, "No components/input_variables is selected!" );
    } else if (type == CV_32SC1) {
        // idx_arr is array of integer indices of selected components
        if( idx_total > data_arr_size )
            CV_ERROR( CV_StsOutOfRange,
            "index array may not contain more elements than the total number of input variables" );
        idx_selected = idx_total;
        // check if sorted already
        for( i = 0; i < idx_total; i++ )
        {
            int val = srci[i*step];
            if( val >= prev )
            {
                is_sorted = 0;
                break;
            }
            prev = val;
        }
    } else {
        CV_ERROR( CV_StsUnsupportedFormat, "Unsupported index array data type "
                                           "(it should be 8uC1, 8sC1 or 32sC1)" );
    }

    CV_CALL( idx = cvCreateMat( 1, idx_selected, CV_32SC1 ));
    dsti = idx->data.i;

    if( type < CV_32SC1 )
    {
        for( i = 0; i < idx_total; i++ )
            if( srcb[i*step] )
                *dsti++ = i;
    }
    else
    {
        for( i = 0; i < idx_total; i++ )
            dsti[i] = srci[i*step];

        if( !is_sorted )
            qsort( dsti, idx_total, sizeof(dsti[0]), icvCmpIntegers );

        if( dsti[0] < 0 || dsti[idx_total-1] >= data_arr_size )
            CV_ERROR( CV_StsOutOfRange, "the index array elements are out of range" );

        if( check_for_duplicates )
        {
            for( i = 1; i < idx_total; i++ )
                if( dsti[i] <= dsti[i-1] )
                    CV_ERROR( CV_StsBadArg, "There are duplicated index array elements" );
        }
    }

    __END__;

    if( cvGetErrStatus() < 0 )
        cvReleaseMat( &idx );

    return idx;
}

//------------------------------------- FeatureEvaluator ---------------------------------------

void FeatureEvaluator::init(Representation *_representation, int maxSampleCount)
{
    representation = _representation;
    cls.create(maxSampleCount, 1, CV_32FC1);
}

void FeatureEvaluator::setImage(const Template &src, uchar clsLabel, int idx)
{
    cls.ptr<float>(idx)[0] = clsLabel;
    data.append(representation->preprocess(src));
}

//----------------------------- CascadeBoostParams -------------------------------------------------

CascadeBoostParams::CascadeBoostParams() : minHitRate( 0.995F), maxFalseAlarm( 0.5F )
{
    boost_type = CvBoost::GENTLE;
    use_surrogates = use_1se_rule = truncate_pruned_tree = false;
}

CascadeBoostParams::CascadeBoostParams(int _boostType,
        float _minHitRate, float _maxFalseAlarm,
        double _weightTrimRate, int _maxDepth, int _maxWeakCount ) :
    CvBoostParams( _boostType, _maxWeakCount, _weightTrimRate, _maxDepth, false, 0 )
{
    boost_type = _boostType;
    minHitRate = _minHitRate;
    maxFalseAlarm = _maxFalseAlarm;
    use_surrogates = use_1se_rule = truncate_pruned_tree = false;
}

//---------------------------- CascadeBoostTrainData -----------------------------

struct CascadeBoostTrainData : CvDTreeTrainData
{
    CascadeBoostTrainData(const FeatureEvaluator* _featureEvaluator, int _channels, const CvDTreeParams& _params);
    CascadeBoostTrainData(const FeatureEvaluator* _featureEvaluator,
                          int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize, int _channels,
                          const CvDTreeParams& _params = CvDTreeParams());
    virtual void setData(const FeatureEvaluator* _featureEvaluator,
                         int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize,
                         const CvDTreeParams& _params=CvDTreeParams());

    void initVarType();
    void precalculate();

    virtual CvDTreeNode* subsample_data(const CvMat* _subsample_idx);

    inline uint64_t getLength() const
    {
        return ((uint64_t)work_var_count + 1) * sample_count;
    }

    virtual const int* getBufferValues(CvDTreeNode* n, uint64_t vi);
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

CvDTreeNode* CascadeBoostTrainData::subsample_data( const CvMat* _subsample_idx )
{
    CvDTreeNode* root = 0;
    CvMat* isubsample_idx = 0;
    CvMat* subsample_co = 0;

    bool isMakeRootCopy = true;

    if( !data_root )
        CV_Error( CV_StsError, "No training data has been set" );

    if( _subsample_idx )
    {
        CV_Assert( (isubsample_idx = cvPreprocessIndexArray( _subsample_idx, sample_count, channels )) != 0 );

        if( isubsample_idx->cols + isubsample_idx->rows - 1 == sample_count )
        {
            const int* sidx = isubsample_idx->data.i;
            for( int i = 0; i < sample_count; i++ )
            {
                if( sidx[i] != i )
                {
                    isMakeRootCopy = false;
                    break;
                }
            }
        }
        else
            isMakeRootCopy = false;
    }

    if( isMakeRootCopy )
    {
        // make a copy of the root node
        CvDTreeNode temp;
        int i;
        root = new_node( 0, 1, 0, 0 );
        temp = *root;
        *root = *data_root;
        root->num_valid = temp.num_valid;
        if( root->num_valid )
        {
            for( i = 0; i < var_count; i++ )
                root->num_valid[i] = data_root->num_valid[i];
        }
        root->cv_Tn = temp.cv_Tn;
        root->cv_node_risk = temp.cv_node_risk;
        root->cv_node_error = temp.cv_node_error;
    }
    else
    {
        int* sidx = isubsample_idx->data.i;
        // co - array of count/offset pairs (to handle duplicated values in _subsample_idx)
        int* co, cur_ofs = 0;
        uint64_t workVarCount = get_work_var_count();
        int count = isubsample_idx->rows + isubsample_idx->cols - 1;

        root = new_node( 0, count, 1, 0 );

        CV_Assert( (subsample_co = cvCreateMat( 1, sample_count*2, CV_32SC1 )) != 0);
        cvZero( subsample_co );
        co = subsample_co->data.i;
        for( int i = 0; i < count; i++ )
            co[sidx[i]*2]++;
        for( int i = 0; i < sample_count; i++ )
        {
            if( co[i*2] )
            {
                co[i*2+1] = cur_ofs;
                cur_ofs += co[i*2];
            }
            else
                co[i*2+1] = -1;
        }

        cv::AutoBuffer<uchar> inn_buf(sample_count*(2*sizeof(int) + sizeof(float)));
        // subsample ordered variables
        for( int vi = 0; vi < numPrecalcIdx; vi++ )
        {
            int ci = get_var_type(vi);
            CV_Assert( ci < 0 );

            int *src_idx_buf = (int*)(uchar*)inn_buf;
            float *src_val_buf = (float*)(src_idx_buf + sample_count);
            int* sample_indices_buf = (int*)(src_val_buf + sample_count);
            const int* src_idx = 0;
            const float* src_val = 0;
            get_ord_var_data( data_root, vi, src_val_buf, src_idx_buf, &src_val, &src_idx, sample_indices_buf );

            int j = 0, idx, count_i;
            int num_valid = data_root->get_num_valid(vi);
            CV_Assert( num_valid == sample_count );

            int* idst_idx = buf->data.i + root->buf_idx*getLength() + (uint64_t)vi*sample_count + root->offset;
            for( int i = 0; i < num_valid; i++ )
            {
                idx = src_idx[i];
                count_i = co[idx*2];
                if( count_i )
                    for( cur_ofs = co[idx*2+1]; count_i > 0; count_i--, j++, cur_ofs++ )
                        idst_idx[j] = cur_ofs;
            }
        }

        // subsample cv_lables
        const int* src_lbls = get_cv_labels(data_root, (int*)(uchar*)inn_buf);
        int* idst = buf->data.i + root->buf_idx*getLength() +
            (workVarCount-1)*sample_count + root->offset;
        for( int i = 0; i < count; i++ )
            idst[i] = src_lbls[sidx[i]];

        // subsample sample_indices
        const int* sample_idx_src = get_sample_indices(data_root, (int*)(uchar*)inn_buf);
        int* sample_idx_dst = buf->data.i + root->buf_idx*getLength() + workVarCount*sample_count + root->offset;
        for( int i = 0; i < count; i++ )
            sample_idx_dst[i] = sample_idx_src[sidx[i]];

        for( int vi = 0; vi < var_count; vi++ )
            root->set_num_valid(vi, count);
    }

    cvReleaseMat( &isubsample_idx );
    cvReleaseMat( &subsample_co );

    return root;
}

CascadeBoostTrainData::CascadeBoostTrainData(const FeatureEvaluator* _featureEvaluator,
                                              int _channels,
                                              const CvDTreeParams& _params)
{
    is_classifier = true;
    var_all = var_count = (int)_featureEvaluator->getNumFeatures();

    featureEvaluator = _featureEvaluator;
    channels = _channels;
    shared = true;
    set_params( _params );
    max_c_count = MAX( 2, featureEvaluator->getMaxCatCount() );

    initVarType();

    int maxSplitSize = cvAlign(sizeof(CvDTreeSplit) + (MAX(0,max_c_count - 33)/32)*sizeof(int),sizeof(void*));
    int treeBlockSize = MAX((int)sizeof(CvDTreeNode)*8, maxSplitSize);
    treeBlockSize = MAX(treeBlockSize + BlockSizeDelta, MinBlockSize);
    tree_storage = cvCreateMemStorage( treeBlockSize );
    node_heap = cvCreateSet( 0, sizeof(node_heap[0]), sizeof(CvDTreeNode), tree_storage );
    split_heap = cvCreateSet( 0, sizeof(split_heap[0]), maxSplitSize, tree_storage );
}

void CascadeBoostTrainData::initVarType()
{
    var_type = cvCreateMat( 1, var_count + 2, CV_32SC1 );
    if ( featureEvaluator->getMaxCatCount() > 0 )
    {
        numPrecalcIdx = 0;
        cat_var_count = var_count;
        ord_var_count = 0;
        for( int vi = 0; vi < var_count; vi++ )
        {
            var_type->data.i[vi] = vi;
        }
    }
    else
    {
        cat_var_count = 0;
        ord_var_count = var_count;
        for( int vi = 1; vi <= var_count; vi++ )
        {
            var_type->data.i[vi-1] = -vi;
        }
    }
    var_type->data.i[var_count] = cat_var_count;
    var_type->data.i[var_count+1] = cat_var_count+1;
}

CascadeBoostTrainData::CascadeBoostTrainData(const FeatureEvaluator* _featureEvaluator,
                                              int _numSamples,
                                              int _precalcValBufSize, int _precalcIdxBufSize, int _channels ,
                                              const CvDTreeParams& _params)
{
    channels = _channels;
    setData( _featureEvaluator, _numSamples, _precalcValBufSize, _precalcIdxBufSize, _params );
}

void CascadeBoostTrainData::setData( const FeatureEvaluator* _featureEvaluator,
                                     int _numSamples,
                                     int _precalcValBufSize, int _precalcIdxBufSize,
                                     const CvDTreeParams& _params )
{
    int* idst = 0;
    unsigned short* udst = 0;

    uint64_t effective_buf_size = 0;
    uint64_t effective_buf_height = 0, effective_buf_width = 0;

    clear();
    shared = true;
    have_labels = true;
    have_priors = false;
    is_classifier = true;

    rng = &cv::theRNG();

    set_params( _params );

    CV_Assert( _featureEvaluator );
    featureEvaluator = _featureEvaluator;

    max_c_count = MAX( 2, featureEvaluator->getMaxCatCount() );
    _resp = featureEvaluator->getCls();
    responses = &_resp;
    // TODO: check responses: elements must be 0 or 1

    if( _precalcValBufSize < 0 || _precalcIdxBufSize < 0)
        CV_Error( CV_StsOutOfRange, "_numPrecalcVal and _numPrecalcIdx must be positive or 0" );

    var_count = var_all = featureEvaluator->getNumFeatures() * featureEvaluator->getFeatureSize();
    sample_count = _numSamples;

    is_buf_16u = false;

    // 1048576 is the number of bytes in a megabyte
    numPrecalcVal = min( cvRound((double)_precalcValBufSize*1048576. / (sizeof(float)*sample_count)), var_count );
    numPrecalcIdx = min( cvRound((double)_precalcIdxBufSize*1048576. / (sizeof (int)*sample_count)), var_count );

    assert( numPrecalcIdx >= 0 && numPrecalcVal >= 0 );

    valCache.create( numPrecalcVal, sample_count, CV_32FC1 );
    var_type = cvCreateMat( 1, var_count + 2, CV_32SC1 );

    if ( featureEvaluator->getMaxCatCount() > 0 )
    {
        numPrecalcIdx = 0;
        cat_var_count = var_count;
        ord_var_count = 0;
        for( int vi = 0; vi < var_count; vi++ )
        {
            var_type->data.i[vi] = vi;
        }
    }
    else
    {
        cat_var_count = 0;
        ord_var_count = var_count;
        for( int vi = 1; vi <= var_count; vi++ )
        {
            var_type->data.i[vi-1] = -vi;
        }
    }
    var_type->data.i[var_count] = cat_var_count;
    var_type->data.i[var_count+1] = cat_var_count+1;

    initVarType();

    work_var_count = ( cat_var_count ? 0 : numPrecalcIdx ) + 1/*cv_lables*/;
    buf_count = 2;

    buf_size = -1; // the member buf_size is obsolete

    effective_buf_size = getLength()*buf_count;

    effective_buf_width = sample_count;
    effective_buf_height = work_var_count+1;
    if (effective_buf_width >= effective_buf_height)
        effective_buf_height *= buf_count;
    else
        effective_buf_width *= buf_count;

    if (effective_buf_width * effective_buf_height != effective_buf_size)
        CV_Error(CV_StsBadArg, "The memory buffer cannot be allocated since its size exceeds integer fields limit");

    buf = cvCreateMat( effective_buf_height, effective_buf_width, CV_32SC1 );

    cat_count = cvCreateMat( 1, cat_var_count + 1, CV_32SC1 );

    // precalculate valCache and set indices in buf
    precalculate();

    // now calculate the maximum size of split,
    // create memory storage that will keep nodes and splits of the decision tree
    // allocate root node and the buffer for the whole training data
    int maxSplitSize = cvAlign(sizeof(CvDTreeSplit) + (MAX(0,sample_count - 33)/32)*sizeof(int),sizeof(void*));
    int treeBlockSize = MAX((int)sizeof(CvDTreeNode)*8, maxSplitSize);
    treeBlockSize = MAX(treeBlockSize + BlockSizeDelta, MinBlockSize);
    tree_storage = cvCreateMemStorage( treeBlockSize );
    node_heap = cvCreateSet( 0, sizeof(*node_heap), sizeof(CvDTreeNode), tree_storage );

    int nvSize = var_count*sizeof(int);
    nvSize = cvAlign(MAX( nvSize, (int)sizeof(CvSetElem) ), sizeof(void*));
    int tempBlockSize = nvSize;
    tempBlockSize = MAX( tempBlockSize + BlockSizeDelta, MinBlockSize );
    temp_storage = cvCreateMemStorage( tempBlockSize );
    nv_heap = cvCreateSet( 0, sizeof(*nv_heap), nvSize, temp_storage );

    data_root = new_node( 0, sample_count, 0, 0 );

    // set sample labels
    idst = buf->data.i + (uint64_t)work_var_count*sample_count;

    for (int si = 0; si < sample_count; si++)
    {
        if (udst)
            udst[si] = (unsigned short)si;
        else
            idst[si] = si;
    }
    for( int vi = 0; vi < var_count; vi++ )
        data_root->set_num_valid(vi, sample_count);
    for( int vi = 0; vi < cat_var_count; vi++ )
        cat_count->data.i[vi] = max_c_count;

    cat_count->data.i[cat_var_count] = 2;

    maxSplitSize = cvAlign(sizeof(CvDTreeSplit) +
        (MAX(0,max_c_count - 33)/32)*sizeof(int),sizeof(void*));
    split_heap = cvCreateSet( 0, sizeof(*split_heap), maxSplitSize, tree_storage );

    priors = cvCreateMat( 1, get_num_classes(), CV_64F );
    cvSet(priors, cvScalar(1));
    priors_mult = cvCloneMat( priors );
    counts = cvCreateMat( 1, get_num_classes(), CV_32SC1 );
    direction = cvCreateMat( 1, sample_count, CV_8UC1 );
    split_buf = cvCreateMat( 1, sample_count, CV_32SC1 );//TODO: make a pointer
}

void CascadeBoostTrainData::free_train_data()
{
    CvDTreeTrainData::free_train_data();
    valCache.release();
}

const int* CascadeBoostTrainData::get_class_labels( CvDTreeNode* n, int* labelsBuf)
{
    int nodeSampleCount = n->sample_count;
    int rStep = CV_IS_MAT_CONT( responses->type ) ? 1 : responses->step / CV_ELEM_SIZE( responses->type );

    int* sampleIndicesBuf = labelsBuf;
    const int* sampleIndices = get_sample_indices(n, sampleIndicesBuf);
    for( int si = 0; si < nodeSampleCount; si++ )
    {
        int sidx = sampleIndices[si];
        labelsBuf[si] = (int)responses->data.fl[sidx*rStep];
    }
    return labelsBuf;
}

const int* CascadeBoostTrainData::getBufferValues(CvDTreeNode* n, uint64_t vi)
{
    return buf->data.i + n->buf_idx*getLength() + vi*sample_count + n->offset;
}

const int* CascadeBoostTrainData::get_sample_indices(CvDTreeNode* n, int* indicesBuf)
{
    Q_UNUSED(indicesBuf)
    return getBufferValues(n,get_work_var_count());
}

const int* CascadeBoostTrainData::get_cv_labels(CvDTreeNode* n, int* indicesBuf)
{
    Q_UNUSED(indicesBuf)
    return getBufferValues(n,get_work_var_count()-1);
}

void CascadeBoostTrainData::get_ord_var_data( CvDTreeNode* n, int vi, float* ordValuesBuf, int* sortedIndicesBuf, const float** ordValues, const int** sortedIndices, int* sampleIndicesBuf )
{
    int nodeSampleCount = n->sample_count;

    // For this node, get our sample indices
    const int* sampleIndices = get_sample_indices(n, sampleIndicesBuf);

    // For this feature (this code refers to features as values, hence vi == value index),
    // have we precalculated (presorted) the training samples by their feature response?
    if (vi < numPrecalcIdx) {
        *sortedIndices = getBufferValues(n, vi);

        // For this feature, have we precalculated all of the feature responses?
        if (vi < numPrecalcVal) {
            float *p = valCache.ptr<float>(vi);
            for (int i = 0; i < nodeSampleCount; i++)
                ordValuesBuf[i] = p[sampleIndices[(*sortedIndices)[i]]];
        } else {
            for (int i = 0; i < nodeSampleCount; i++) {
                int idx = (*sortedIndices)[i];
                idx = sampleIndices[idx];
                ordValuesBuf[i] = (*featureEvaluator)(vi, idx);
            }
        }
    } else {
        cv::AutoBuffer<float> abuf(nodeSampleCount);
        float* sampleValues = &abuf[0];

        if (vi < numPrecalcVal) {
            for (int i = 0; i < nodeSampleCount; i++) {
                sortedIndicesBuf[i] = i;
                sampleValues[i] = valCache.at<float>( vi, sampleIndices[i] );
            }
        } else {
            for (int i = 0; i < nodeSampleCount; i++) {
                sortedIndicesBuf[i] = i;
                sampleValues[i] = (*featureEvaluator)( vi, sampleIndices[i]);
            }
        }

        icvSortIntAux( sortedIndicesBuf, nodeSampleCount, &sampleValues[0] );

        for (int i = 0; i < nodeSampleCount; i++)
            ordValuesBuf[i] = (&sampleValues[0])[sortedIndicesBuf[i]];

        *sortedIndices = sortedIndicesBuf;
    }

    *ordValues = ordValuesBuf;
}

const int* CascadeBoostTrainData::get_cat_var_data( CvDTreeNode* n, int vi, int* catValuesBuf )
{
    int nodeSampleCount = n->sample_count;
    int* sampleIndicesBuf = catValuesBuf;

    const int* sampleIndices = get_sample_indices(n, sampleIndicesBuf);

    if ( vi < numPrecalcVal )
    {
        for( int i = 0; i < nodeSampleCount; i++ )
            catValuesBuf[i] = (int) valCache.at<float>( vi, sampleIndices[i]);
    }
    else
    {
        if( vi >= numPrecalcVal && vi < var_count )
        {
            for( int i = 0; i < nodeSampleCount; i++ )
                catValuesBuf[i] = (int)(*featureEvaluator)( vi, sampleIndices[i] );
        }
        else
        {
            get_cv_labels( n, catValuesBuf );
        }
    }

    return catValuesBuf;
}

float CascadeBoostTrainData::getVarValue( int vi, int si )
{
    if ( vi < numPrecalcVal && !valCache.empty() )
        return valCache.at<float>( vi, si );
    return (*featureEvaluator)( vi, si );
}

struct Precalc : ParallelLoopBody
{
    const FeatureEvaluator* featureEvaluator;
    int sampleCount;

    Precalc(const FeatureEvaluator* featureEvaluator, int sampleCount) :
        featureEvaluator(featureEvaluator),
        sampleCount(sampleCount)
    {}

    virtual void operator()(const Range& range) const = 0;
};

struct IndexPrecalc : Precalc
{
    int* idst;
    unsigned short* udst;

    IndexPrecalc(const FeatureEvaluator* featureEvaluator, CvMat* buf, int sampleCount) :
        Precalc(featureEvaluator, sampleCount),
        idst(buf->data.i)
    {}

    void setBuffer(uint64_t fi, uint64_t si) const
    {
        *(idst + fi*sampleCount + si) = si;
    }

    void sortBuffer(uint64_t fi, float *valCachePtr) const
    {
        icvSortIntAux(idst + fi*sampleCount, sampleCount, valCachePtr);
    }

    virtual void operator()(const Range& range) const
    {
        cv::AutoBuffer<float> valCache(sampleCount);
        float* valCachePtr = (float*)valCache;
        for (int fi = range.start; fi < range.end; fi++) {
            for (int si = 0; si < sampleCount; si++) {
                valCachePtr[si] = (*featureEvaluator)(fi, si);
                setBuffer(fi, si);
            }
            sortBuffer(fi, valCachePtr);
        }
    }
};

struct FeatureAndIndexPrecalc : IndexPrecalc
{
    Mat *valCache;

    FeatureAndIndexPrecalc(const FeatureEvaluator* featureEvaluator, CvMat* buf, Mat* valCache, int sampleCount) :
        IndexPrecalc(featureEvaluator, buf, sampleCount),
        valCache(valCache)
    {}

    virtual void operator()(const Range& range) const
    {
        for (int fi = range.start; fi < range.end; fi++) {
            for (int si = 0; si < sampleCount; si++) {
                valCache->at<float>(fi,si) = (*featureEvaluator)(fi, si);
                setBuffer(fi, si);
            }
            sortBuffer(fi, valCache->ptr<float>(fi));
        }
    }
};

struct FeaturePrecalc : Precalc
{
    Mat *valCache;

    FeaturePrecalc(const FeatureEvaluator* featureEvaluator, Mat* valCache, int sampleCount) :
        Precalc(featureEvaluator, sampleCount),
        valCache(valCache)
    {}

    virtual void operator()(const Range& range) const
    {
        for (int fi = range.start; fi < range.end; fi++)
            for (int si = 0; si < sampleCount; si++)
                valCache->at<float>(fi,si) = (*featureEvaluator)(fi, si);
    }
};

void CascadeBoostTrainData::precalculate()
{
    int minPrecalc = std::min(numPrecalcVal, numPrecalcIdx);

    qDebug() << "Starting precalculation...";

    QTime time;
    time.start();

    // Compute features and sort training samples for feature indices we are not going to cache
    parallel_for_(Range(numPrecalcVal, numPrecalcIdx),
                  IndexPrecalc(featureEvaluator, buf, sample_count));

    // Compute features and sort training samples for features indices we are going to cache
    parallel_for_(Range(0, minPrecalc),
                  FeatureAndIndexPrecalc(featureEvaluator, buf, &valCache, sample_count));

    // Compute feature values for feature indices for which we are not going to sort training samples
    parallel_for_(Range(minPrecalc, numPrecalcVal),
                  FeaturePrecalc(featureEvaluator, &valCache, sample_count));

    cout << "Precalculation time (min): " << time.elapsed()/1000/60. <<  endl;
}

//-------------------------------- CascadeBoostTree ----------------------------------------

CvDTreeNode* CascadeBoostTree::predict( int sampleIdx ) const
{
    CvDTreeNode* node = root;
    if( !node )
        CV_Error( CV_StsError, "The tree has not been trained yet" );

    if ( ((CascadeBoostTrainData*)data)->featureEvaluator->getMaxCatCount() == 0 ) // ordered
    {
        while( node->left )
        {
            CvDTreeSplit* split = node->split;
            float val = ((CascadeBoostTrainData*)data)->getVarValue( split->var_idx, sampleIdx );
            node = val <= split->ord.c ? node->left : node->right;
        }
    }
    else // categorical
    {
        while( node->left )
        {
            CvDTreeSplit* split = node->split;
            int c = (int)((CascadeBoostTrainData*)data)->getVarValue( split->var_idx, sampleIdx );
            node = CV_DTREE_CAT_DIR(c, split->subset) < 0 ? node->left : node->right;
        }
    }
    return node;
}

// This function splits the training data from the parent node into training
// data for both child nodes
void CascadeBoostTree::split_node_data( CvDTreeNode* node )
{
    CvDTreeNode *left = 0, *right = 0;

    uint64_t nodeSampleCount = node->sample_count;
    uint64_t sampleCount = data->sample_count;
    uint64_t nLeft, nRight;

    int workVarCount = data->get_work_var_count();

    char* dir = (char*)data->direction->data.ptr;
    int* newIdx = data->split_buf->data.i;
    int newBufIdx = data->get_child_buf_idx( node );
    CvMat* buf = data->buf;
    size_t length_buf_row = ((CascadeBoostTrainData*)data)->getLength();
    cv::AutoBuffer<uchar> inn_buf(nodeSampleCount*(3*sizeof(int)+sizeof(float)));
    int* tempBuf = (int*)(uchar*)inn_buf;

    complete_node_dir(node);

    for (uint64_t i = nLeft = nRight = 0; i < nodeSampleCount; i++) {
        int d = dir[i];
        // initialize new indices for splitting ordered variables
        newIdx[i] = (nLeft & (d-1)) | (nRight & -d); // d ? ri : li
        nRight += d;
        nLeft += d^1;
    }

    node->left = left = data->new_node( node, nLeft, newBufIdx, node->offset );
    node->right = right = data->new_node( node, nRight, newBufIdx, node->offset + nLeft );

    bool splitInputData = node->depth + 1 < data->params.max_depth && (node->left->sample_count > data->params.min_sample_count || node->right->sample_count > data->params.min_sample_count);

    const int numPreculatedIndices = ((CascadeBoostTrainData*)data)->numPrecalcIdx;
    for (int vi = 0; vi < numPreculatedIndices; vi++) {
        int ci = data->get_var_type(vi);
        if( ci >= 0 || !splitInputData )
            continue;

        const int* src_sorted_idx = ((CascadeBoostTrainData*)data)->getBufferValues(node, vi);
        for(uint64_t i = 0; i < nodeSampleCount; i++)
            tempBuf[i] = src_sorted_idx[i];

        int *ldst, *rdst;
        ldst = buf->data.i + left->buf_idx*length_buf_row + vi*sampleCount + left->offset;
        rdst = buf->data.i + right->buf_idx*length_buf_row + vi*sampleCount + right->offset;

        int n1 = node->get_num_valid(vi);

        // split sorted
        for (int i = 0; i < n1; i++) {
            int idx = tempBuf[i];
            int d = dir[idx];
            idx = newIdx[idx];
            if (d) {
                *rdst = idx;
                rdst++;
            } else {
                *ldst = idx;
                ldst++;
            }
        }
    }

    // split cv_labels using newIdx relocation table
    int *src_lbls_buf = tempBuf + nodeSampleCount;
    const int* src_lbls = data->get_cv_labels(node, src_lbls_buf);

    for(uint64_t i = 0; i < nodeSampleCount; i++)
        tempBuf[i] = src_lbls[i];

    {
        int *ldst = buf->data.i + left->buf_idx*length_buf_row + (workVarCount-1)*sampleCount + left->offset;
        int *rdst = buf->data.i + right->buf_idx*length_buf_row + (workVarCount-1)*sampleCount + right->offset;

        for (uint64_t i = 0; i < nodeSampleCount; i++) {
            int idx = tempBuf[i];
            if (dir[i]) {
                *rdst = idx;
                rdst++;
            } else {
                *ldst = idx;
                ldst++;
            }
        }
    }

    // split sample indices
    int *sampleIdx_src_buf = tempBuf + nodeSampleCount;
    const int* sampleIdx_src = data->get_sample_indices(node, sampleIdx_src_buf);

    for(uint64_t i = 0; i < nodeSampleCount; i++)
        tempBuf[i] = sampleIdx_src[i];

    {
        int* ldst = buf->data.i + left->buf_idx*length_buf_row + workVarCount*sampleCount + left->offset;
        int* rdst = buf->data.i + right->buf_idx*length_buf_row + workVarCount*sampleCount + right->offset;
        for (uint64_t i = 0; i < nodeSampleCount; i++) {
            int idx = tempBuf[i];
            if (dir[i]) {
                *rdst = idx;
                rdst++;
            } else {
                *ldst = idx;
                ldst++;
            }
        }
    }

    const int variableCount = data->var_count;
    for (int vi = 0; vi < variableCount; vi++) {
        left->set_num_valid(vi, nLeft);
        right->set_num_valid(vi, nRight);
    }

    // deallocate the parent node data that is not needed anymore
    data->free_node_data(node);
}

//----------------------------------- CascadeBoost --------------------------------------

void CascadeBoost::train(const FeatureEvaluator* _featureEvaluator,
                         int _numSamples,
                         int _precalcValBufSize,
                         int _precalcIdxBufSize,
                         int _channels,
                         const CascadeBoostParams& _params)
{
    CV_Assert(!data);
    clear();

    channels = _channels;

    data = new CascadeBoostTrainData(_featureEvaluator, _numSamples,
                                        _precalcValBufSize, _precalcIdxBufSize, channels, _params);

    set_params(_params);
    if ((_params.boost_type == LOGIT) || (_params.boost_type == GENTLE))
        data->do_responses_copy();

    update_weights(0);

    cout << "+----+---------+---------+" << endl;
    cout << "|  N |    HR   |    FA   |" << endl;
    cout << "+----+---------+---------+" << endl;

    do {
        CascadeBoostTree* tree = new CascadeBoostTree;
        if (!tree->train( data, subsample_mask, this)) {
            delete tree;
            return;
        }

        classifiers.append(tree);
        update_weights(tree);
        trim_weights();
        if (cvCountNonZero(subsample_mask) == 0) {
            return;
	    }
    }
    while (!isErrDesired() && (classifiers.size() < params.weak_count));

    //clear();
}

float CascadeBoost::predict(int sampleIdx, bool returnSum) const
{
    double sum = 0;
    foreach (const CvBoostTree *tree, classifiers)
        sum += ((CascadeBoostTree*)tree)->predict(sampleIdx)->value;

    if (!returnSum)
        sum = sum < threshold - CV_THRESHOLD_EPS ? 0.0 : 1.0;

    return (float)sum;
}

bool CascadeBoost::set_params(const CvBoostParams& _params)
{
    minHitRate = ((CascadeBoostParams&)_params).minHitRate;
    maxFalseAlarm = ((CascadeBoostParams&)_params).maxFalseAlarm;
    return (( minHitRate > 0 ) && ( minHitRate < 1) &&
            (maxFalseAlarm > 0 ) && ( maxFalseAlarm < 1) &&
            CvBoost::set_params(_params));
}

void CascadeBoost::update_weights(CvBoostTree* tree)
{
    int n = data->sample_count;
    double sumW = 0.;
    int step = 0;
    float* fdata = 0;
    int *sampleIdxBuf;
    const int* sampleIdx = 0;
    int inn_buf_size = ((params.boost_type == LOGIT) || (params.boost_type == GENTLE) ? n*sizeof(int) : 0) +
                       ( !tree ? n*sizeof(int) : 0 );
    cv::AutoBuffer<uchar> inn_buf(inn_buf_size);
    uchar* cur_inn_buf_pos = (uchar*)inn_buf;

    if ( (params.boost_type == LOGIT) || (params.boost_type == GENTLE) )
    {
        step = CV_IS_MAT_CONT(data->responses_copy->type) ?
            1 : data->responses_copy->step / CV_ELEM_SIZE(data->responses_copy->type);
        fdata = data->responses_copy->data.fl;
        sampleIdxBuf = (int*)cur_inn_buf_pos; cur_inn_buf_pos = (uchar*)(sampleIdxBuf + n);
        sampleIdx = data->get_sample_indices( data->data_root, sampleIdxBuf );
    }

    CvMat* buf = data->buf;
    uint64_t length_buf_row = ((CascadeBoostTrainData*)data)->getLength();
    if( !tree ) // before training the first tree, initialize weights and other parameters
    {
        int* classLabelsBuf = (int*)cur_inn_buf_pos; cur_inn_buf_pos = (uchar*)(classLabelsBuf + n);
        const int* classLabels = data->get_class_labels(data->data_root, classLabelsBuf);
        // in case of logitboost and gentle adaboost each weak tree is a regression tree,
        // so we need to convert class labels to floating-point values
        double w0 = 1./n;
        double p[2] = { 1, 1 };

        cvReleaseMat( &orig_response );
        cvReleaseMat( &sum_response );
        cvReleaseMat( &weak_eval );
        cvReleaseMat( &subsample_mask );
        cvReleaseMat( &weights );

        orig_response = cvCreateMat( 1, n, CV_32S );
        weak_eval = cvCreateMat( 1, n, CV_64F );
        subsample_mask = cvCreateMat( 1, n, CV_8U );
        weights = cvCreateMat( 1, n, CV_64F );
        subtree_weights = cvCreateMat( 1, n + 2, CV_64F );

        // set the labels to find (from within weak tree learning proc)
        // the particular sample weight, and where to store the response.
        int* labels = buf->data.i + data->data_root->buf_idx*length_buf_row + data->data_root->offset + (uint64_t)(data->work_var_count-1)*data->sample_count;
        for( int i = 0; i < n; i++ )
            labels[i] = i;

        for (int i = 0; i < n; i++) {
            // save original categorical responses {0,1}, convert them to {-1,1}
            orig_response->data.i[i] = classLabels[i]*2 - 1;
            // make all the samples active at start.
            // later, in trim_weights() deactivate/reactive again some, if need
            subsample_mask->data.ptr[i] = (uchar)1;
            // make all the initial weights the same.
            weights->data.db[i] = w0*p[classLabels[i]];
        }

        if( params.boost_type == LOGIT )
        {
            sum_response = cvCreateMat( 1, n, CV_64F );

            for( int i = 0; i < n; i++ )
            {
                sum_response->data.db[i] = 0;
                fdata[sampleIdx[i]*step] = orig_response->data.i[i] > 0 ? 2.f : -2.f;
            }

            // in case of logitboost each weak tree is a regression tree.
            // the target function values are recalculated for each of the trees
            data->is_classifier = false;
        }
        else if( params.boost_type == GENTLE )
        {
            for( int i = 0; i < n; i++ )
                fdata[sampleIdx[i]*step] = (float)orig_response->data.i[i];

            data->is_classifier = false;
        }
    }
    else
    {
        // at this moment, for all the samples that participated in the training of the most
        // recent weak classifier we know the responses. For other samples we need to compute them
        if( have_subsample )
        {
            // invert the subsample mask
            cvXorS( subsample_mask, cvScalar(1.), subsample_mask );

            // run tree through all the non-processed samples
            for( int i = 0; i < n; i++ )
                if( subsample_mask->data.ptr[i] )
                {
                    weak_eval->data.db[i] = ((CascadeBoostTree*)tree)->predict( i )->value;
                }
        }

        // now update weights and other parameters for each type of boosting
        if( params.boost_type == DISCRETE )
        {
            // Discrete AdaBoost:
            //   weak_eval[i] (=f(x_i)) is in {-1,1}
            //   err = sum(w_i*(f(x_i) != y_i))/sum(w_i)
            //   C = log((1-err)/err)
            //   w_i *= exp(C*(f(x_i) != y_i))

            double C, err = 0.;
            double scale[] = { 1., 0. };

            for( int i = 0; i < n; i++ )
            {
                double w = weights->data.db[i];
                sumW += w;
                err += w*(weak_eval->data.db[i] != orig_response->data.i[i]);
            }

            if( sumW != 0 )
                err /= sumW;
            C = err = -logRatio( err );
            scale[1] = exp(err);

            sumW = 0;
            for( int i = 0; i < n; i++ )
            {
                double w = weights->data.db[i]*
                    scale[weak_eval->data.db[i] != orig_response->data.i[i]];
                sumW += w;
                weights->data.db[i] = w;
            }

            tree->scale( C );
        }
        else if( params.boost_type == REAL )
        {
            // Real AdaBoost:
            //   weak_eval[i] = f(x_i) = 0.5*log(p(x_i)/(1-p(x_i))), p(x_i)=P(y=1|x_i)
            //   w_i *= exp(-y_i*f(x_i))

            for( int i = 0; i < n; i++ )
                weak_eval->data.db[i] *= -orig_response->data.i[i];

            cvExp( weak_eval, weak_eval );

            for( int i = 0; i < n; i++ )
            {
                double w = weights->data.db[i]*weak_eval->data.db[i];
                sumW += w;
                weights->data.db[i] = w;
            }
        }
        else if( params.boost_type == LOGIT )
        {
            // LogitBoost:
            //   weak_eval[i] = f(x_i) in [-z_max,z_max]
            //   sum_response = F(x_i).
            //   F(x_i) += 0.5*f(x_i)
            //   p(x_i) = exp(F(x_i))/(exp(F(x_i)) + exp(-F(x_i))=1/(1+exp(-2*F(x_i)))
            //   reuse weak_eval: weak_eval[i] <- p(x_i)
            //   w_i = p(x_i)*1(1 - p(x_i))
            //   z_i = ((y_i+1)/2 - p(x_i))/(p(x_i)*(1 - p(x_i)))
            //   store z_i to the data->data_root as the new target responses

            const double lbWeightThresh = FLT_EPSILON;
            const double lbZMax = 10.;

            for( int i = 0; i < n; i++ )
            {
                double s = sum_response->data.db[i] + 0.5*weak_eval->data.db[i];
                sum_response->data.db[i] = s;
                weak_eval->data.db[i] = -2*s;
            }

            cvExp( weak_eval, weak_eval );

            for( int i = 0; i < n; i++ )
            {
                double p = 1./(1. + weak_eval->data.db[i]);
                double w = p*(1 - p), z;
                w = MAX( w, lbWeightThresh );
                weights->data.db[i] = w;
                sumW += w;
                if( orig_response->data.i[i] > 0 )
                {
                    z = 1./p;
                    fdata[sampleIdx[i]*step] = (float)min(z, lbZMax);
                }
                else
                {
                    z = 1./(1-p);
                    fdata[sampleIdx[i]*step] = (float)-min(z, lbZMax);
                }
            }
        }
        else
        {
            // Gentle AdaBoost:
            //   weak_eval[i] = f(x_i) in [-1,1]
            //   w_i *= exp(-y_i*f(x_i))
            assert( params.boost_type == GENTLE );

            for( int i = 0; i < n; i++ )
                weak_eval->data.db[i] *= -orig_response->data.i[i];

            cvExp( weak_eval, weak_eval );

            for( int i = 0; i < n; i++ )
            {
                double w = weights->data.db[i] * weak_eval->data.db[i];
                weights->data.db[i] = w;
                sumW += w;
            }
        }
    }

    // renormalize weights
    if( sumW > FLT_EPSILON )
    {
        sumW = 1./sumW;
        for( int i = 0; i < n; ++i )
            weights->data.db[i] *= sumW;
    }
}

bool CascadeBoost::isErrDesired()
{
    QList<float> posVals;
    for (int i = 0; i < data->sample_count; i++)
        if (((CascadeBoostTrainData*)data)->featureEvaluator->getCls(i) == 1.0F)
            posVals.append(predict(i, true));

    std::sort(posVals.begin(), posVals.end());

    int thresholdIdx = (int)((1.0F - minHitRate) * posVals.size());
    threshold = posVals[thresholdIdx];

    int numPosTrue = posVals.size() - thresholdIdx;
    for (int i = thresholdIdx - 1; i >= 0; i--)
        if (abs(posVals[i] - threshold) < FLT_EPSILON)
            numPosTrue++;
    float hitRate = ((float)numPosTrue) / ((float)posVals.size());

    int numNeg = 0, numFalse = 0;
    for (int i = 0; i < data->sample_count; i++) {
        if (((CascadeBoostTrainData*)data)->featureEvaluator->getCls(i) == 0.0F) {
            numNeg++;
            if (predict(i))
                numFalse++;
        }
    }
    float falseAlarm = ((float)numFalse) / ((float)numNeg);

    cout << "|"; cout.width(4); cout << right << classifiers.size();
    cout << "|"; cout.width(9); cout << right << hitRate;
    cout << "|"; cout.width(9); cout << right << falseAlarm;
    cout << "|" << endl;
    cout << "+----+---------+---------+" << endl;

    return falseAlarm <= maxFalseAlarm;
}
