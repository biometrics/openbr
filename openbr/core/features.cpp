#include "features.h"

using namespace cv;
using namespace br;

//------------------------- Params -----------------------------------------------

float calcNormFactor( const Mat& sum, const Mat& sqSum )
{
    CV_DbgAssert( sum.cols > 3 && sqSum.rows > 3 );
    Rect normrect( 1, 1, sum.cols - 3, sum.rows - 3 );
    size_t p0, p1, p2, p3;
    CV_SUM_OFFSETS( p0, p1, p2, p3, normrect, sum.step1() )
    double area = normrect.width * normrect.height;
    const int *sp = (const int*)sum.data;
    int valSum = sp[p0] - sp[p1] - sp[p2] + sp[p3];
    const double *sqp = (const double *)sqSum.data;
    double valSqSum = sqp[p0] - sqp[p1] - sqp[p2] + sqp[p3];
    return (float) sqrt( (double) (area * valSqSum - (double)valSum * valSum) );
}

Params::Params() : name( "params" ) {}
void Params::printDefaults() const { std::cout << "--" << name << "--" << endl; }
void Params::printAttrs() const {}
bool Params::scanAttr( const string, const string ) { return false; }


//---------------------------- FeatureParams --------------------------------------

FeatureParams::FeatureParams() : maxCatCount( 0 ), featSize( 1 )
{
    name = CC_FEATURE_PARAMS;
}

void FeatureParams::init( const FeatureParams& fp )
{
    maxCatCount = fp.maxCatCount;
    featSize = fp.featSize;
}

void FeatureParams::write( FileStorage &fs ) const
{
    fs << CC_MAX_CAT_COUNT << maxCatCount;
    fs << CC_FEATURE_SIZE << featSize;
}

bool FeatureParams::read( const FileNode &node )
{
    if ( node.empty() )
        return false;
    maxCatCount = node[CC_MAX_CAT_COUNT];
    featSize = node[CC_FEATURE_SIZE];
    return ( maxCatCount >= 0 && featSize >= 1 );
}

Ptr<FeatureParams> FeatureParams::create( int featureType )
{
    return featureType == LBP ? Ptr<FeatureParams>(new LBPFeatureParams) :
                                Ptr<FeatureParams>();
}

//------------------------------------- FeatureEvaluator ---------------------------------------

void FeatureEvaluator::init(const FeatureParams *_featureParams,
                              int _maxSampleCount, Size _winSize )
{
    CV_Assert(_maxSampleCount > 0);
    featureParams = (FeatureParams *)_featureParams;
    winSize = _winSize;
    numFeatures = 0;
    cls.create( (int)_maxSampleCount, 1, CV_32FC1 );
    generateFeatures();
}

void FeatureEvaluator::setImage(const Mat &img, uchar clsLabel, int idx)
{
    CV_Assert(img.cols == winSize.width);
    CV_Assert(img.rows == winSize.height);
    CV_Assert(idx < cls.rows);
    cls.ptr<float>(idx)[0] = clsLabel;
}

Ptr<FeatureEvaluator> FeatureEvaluator::create(int type)
{
    return type == FeatureParams::LBP ? Ptr<FeatureEvaluator>(new LBPEvaluator) :
                                        Ptr<FeatureEvaluator>();
}

// ------------------------------------ LBP -----------------------------------------------

LBPFeatureParams::LBPFeatureParams()
{
    maxCatCount = 256;
    name = LBPF_NAME;
}

void LBPEvaluator::init(const FeatureParams *_featureParams, int _maxSampleCount, Size _winSize)
{
    CV_Assert( _maxSampleCount > 0);
    sum.create((int)_maxSampleCount, (_winSize.width + 1) * (_winSize.height + 1), CV_32SC1);
    FeatureEvaluator::init( _featureParams, _maxSampleCount, _winSize );
}

void LBPEvaluator::setImage(const Mat &img, uchar clsLabel, int idx)
{
    CV_DbgAssert( !sum.empty() );
    FeatureEvaluator::setImage( img, clsLabel, idx );
    Mat innSum(winSize.height + 1, winSize.width + 1, sum.type(), sum.ptr<int>((int)idx));
    integral( img, innSum );
}

void LBPEvaluator::writeFeatures( FileStorage &fs, const Mat& featureMap ) const
{
    _writeFeatures( features, fs, featureMap );
}

void LBPEvaluator::generateFeatures()
{
    int offset = winSize.width + 1;
    for( int x = 0; x < winSize.width; x++ )
        for( int y = 0; y < winSize.height; y++ )
            for( int w = 1; w <= winSize.width / 3; w++ )
                for( int h = 1; h <= winSize.height / 3; h++ )
                    if ( (x+3*w <= winSize.width) && (y+3*h <= winSize.height) )
                        features.push_back( Feature(offset, x, y, w, h ) );
    numFeatures = (int)features.size();
}

LBPEvaluator::Feature::Feature()
{
    rect = cvRect(0, 0, 0, 0);
}

LBPEvaluator::Feature::Feature( int offset, int x, int y, int _blockWidth, int _blockHeight )
{
    Rect tr = rect = cvRect(x, y, _blockWidth, _blockHeight);
    CV_SUM_OFFSETS( p[0], p[1], p[4], p[5], tr, offset )
    tr.x += 2*rect.width;
    CV_SUM_OFFSETS( p[2], p[3], p[6], p[7], tr, offset )
    tr.y +=2*rect.height;
    CV_SUM_OFFSETS( p[10], p[11], p[14], p[15], tr, offset )
    tr.x -= 2*rect.width;
    CV_SUM_OFFSETS( p[8], p[9], p[12], p[13], tr, offset )
}

void LBPEvaluator::Feature::write(FileStorage &fs) const
{
    fs << CC_RECT << "[:" << rect.x << rect.y << rect.width << rect.height << "]";
}
